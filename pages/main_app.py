import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import colorsys
import os
import json
import numpy as np
from scipy.stats import norm as _norm
import pydeck as pdk
import base64
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, "db", "users.db")

# ── Fertilizer constants (nitrogen intensity per crop) ────────────────────────
N_INTENSITY_LBS_PER_ACRE = {
    "Corn": 150, "Wheat": 90, "Cotton": 120,
    "Sorghum": 80, "Soybeans": 60, "Hay": 50,
}
UREA_N_CONTENT = 0.46   # urea is 46% nitrogen by weight
LBS_PER_MT     = 2204.6 # pounds per metric ton

# USDA 2023 planted acres by state (thousands of acres)
# (lat, lng, corn_k, wheat_k, soy_k)
_STATE_AG = {
    "Iowa":           (42.00, -93.50, 12900,     0,  9400),
    "Illinois":       (40.00, -89.00, 10800,   500, 10100),
    "Nebraska":       (41.50, -99.90, 10300,  1400,  5500),
    "Minnesota":      (46.40, -94.30,  8100,  1600,  7500),
    "Indiana":        (40.30, -86.10,  5400,   500,  5800),
    "South Dakota":   (44.30,-100.30,  5200,  1600,  4300),
    "Kansas":         (38.50, -98.40,  4400,  7800,  5000),
    "Ohio":           (40.40, -82.70,  3700,   500,  4900),
    "Wisconsin":      (44.50, -89.50,  3600,   100,  1900),
    "Missouri":       (38.30, -92.40,  3100,   600,  5400),
    "North Dakota":   (47.50,-100.50,  2600,  5500,  5900),
    "Michigan":       (44.30, -84.50,  2300,   400,  2300),
    "Texas":          (31.40, -99.30,  1900,  5000,   100),
    "Colorado":       (39.00,-105.50,  1500,  2500,   400),
    "Kentucky":       (37.80, -84.90,  1300,   400,  1600),
    "Oklahoma":       (35.60, -97.50,   400,  4500,   700),
    "Montana":        (46.90,-110.40,   200,  4500,   100),
    "North Carolina": (35.50, -79.30,   900,   100,  1100),
    "Arkansas":       (34.80, -92.20,   800,   200,  3000),
    "Tennessee":      (35.80, -86.50,   800,   200,  1500),
    "Pennsylvania":   (41.20, -77.20,  1100,   200,   700),
    "New York":       (42.50, -76.00,   800,   200,   300),
    "Mississippi":    (32.70, -89.70,   400,   100,  2300),
    "Washington":     (47.40,-120.40,   100,  2100,   100),
    "Virginia":       (37.40, -79.00,   400,   200,   600),
    "Georgia":        (32.20, -83.40,   500,   300,   500),
    "Maryland":       (39.00, -76.80,   500,   200,   300),
    "Idaho":          (44.20,-114.50,   100,  1500,   100),
    "Wyoming":        (43.00,-107.50,   100,   500,   100),
    "California":     (37.00,-119.50,   200,   700,   100),
    "Delaware":       (39.00, -75.50,   200,   100,   200),
}

# ── Auth guard ───────────────────────────────────────────────────────────────
# On reload session_state is cleared — restore from the ?s= token in the URL
if not st.session_state.get("logged_in"):
    _tok = st.query_params.get("s")
    if _tok:
        _conn = sqlite3.connect(DB_PATH)
        _row  = _conn.execute(
            "SELECT email FROM session_tokens WHERE token = ? AND expires_at > ?",
            (_tok, int(time.time())),
        ).fetchone()
        _conn.close()
        if _row:
            st.session_state.logged_in         = True
            st.session_state.user_email        = _row[0]
            st.session_state["_session_token"] = _tok

if not st.session_state.get("logged_in"):
    st.switch_page("pages/login.py")

# Keep token in URL so F5 refresh restores the session
if "s" not in st.query_params and st.session_state.get("_session_token"):
    st.query_params["s"] = st.session_state["_session_token"]

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "cache.json")


def load_cache():
    """Safely loads the generated ML cache file. Returns None if unavailable."""
    try:
        with open(CACHE_PATH, "r") as f:
            data = json.load(f)
        # Validate required top-level keys are present
        required = {"signal", "monte_carlo", "forecast", "model_metadata"}
        if not required.issubset(data.keys()):
            print(f"Warning: cache.json is missing keys: {required - data.keys()}")
            return None
        return data
    except Exception as e:
        print(f"Warning: Could not load cache.json: {e}")
        return None

# ── Database ─────────────────────────────────────────────────────────────────
def init_farm_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS farm_crops (
            id         INTEGER PRIMARY KEY,
            user_email TEXT    NOT NULL,
            crop_name  TEXT    NOT NULL,
            acres      REAL    NOT NULL,
            season     TEXT    NOT NULL DEFAULT 'Spring',
            year       INTEGER NOT NULL DEFAULT 2025
        )
    """)
    # Migrate existing tables that may lack season or year columns
    cols = [r[1] for r in conn.execute("PRAGMA table_info(farm_crops)").fetchall()]
    if "season" not in cols:
        conn.execute("ALTER TABLE farm_crops ADD COLUMN season TEXT NOT NULL DEFAULT 'Spring'")
    if "year" not in cols:
        conn.execute("ALTER TABLE farm_crops ADD COLUMN year INTEGER NOT NULL DEFAULT 2025")
    conn.commit()
    conn.close()


def get_farmer_crops(email: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id, crop_name, acres, season, year FROM farm_crops WHERE user_email = ?",
        conn, params=(email,),
    )
    conn.close()
    return df


def add_crop(email: str, crop_name: str, acres: float, season: str, year: int):
    """Insert or merge crop. Same crop name (case-insensitive) + same season + same year → add acres."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT id, crop_name FROM farm_crops WHERE user_email=? AND LOWER(crop_name)=LOWER(?) AND season=? AND year=?",
        (email, crop_name, season, year),
    ).fetchone()
    if row:
        conn.execute("UPDATE farm_crops SET acres = acres + ? WHERE id = ?", (acres, row[0]))
    else:
        conn.execute(
            "INSERT INTO farm_crops (user_email, crop_name, acres, season, year) VALUES (?,?,?,?,?)",
            (email, crop_name, acres, season, year),
        )
    conn.commit()
    conn.close()


def update_crop_acres(row_id: int, new_acres: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE farm_crops SET acres = ? WHERE id = ?", (new_acres, row_id))
    conn.commit()
    conn.close()


def delete_crop(row_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM farm_crops WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()


# ── Image helper ──────────────────────────────────────────────────────────────
def get_image_base64(filename):
    """Convert image to base64 data URL"""
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", filename)
    if os.path.exists(img_path):
        with open(img_path, "rb") as img_file:
            data = base64.b64encode(img_file.read()).decode()
            ext = filename.split(".")[-1].lower()
            mime_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"
            return f"data:{mime_type};base64,{data}"
    return None


# ── Season helpers ────────────────────────────────────────────────────────────
DEFAULT_SEASONS = ["Spring", "Fall"]

# Base hue for each season (hue in 0-1 range for colorsys)
SEASON_BASE_COLORS = {"Spring": "#2ecc71", "Fall": "#e67e22"}
EXTRA_PALETTE = ["#3498db", "#9b59b6", "#e74c3c", "#1abc9c", "#f39c12", "#34495e"]


def get_season_base_colors(seasons: list) -> dict:
    """Maps each season name to a base hex color."""
    result = {}
    extra_idx = 0
    for s in seasons:
        if s in SEASON_BASE_COLORS:
            result[s] = SEASON_BASE_COLORS[s]
        else:
            result[s] = EXTRA_PALETTE[extra_idx % len(EXTRA_PALETTE)]
            extra_idx += 1
    return result


def _hex_to_hls(hex_color: str):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h, lightness, s) -> str:
    r, g, b = colorsys.hls_to_rgb(h, lightness, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# Fixed alternating pattern: dark → light → medium-dark → medium-light → darkest → medium
# Applied as (lightness, saturation_scale) pairs, cycling for any number of crops
_SHADE_PATTERN = [
    (0.34, 1.00),
    (0.65, 0.70),
    (0.44, 0.90),
    (0.72, 0.55),
    (0.26, 1.00),
    (0.55, 0.80),
]


def get_crop_colors_for_df(df: pd.DataFrame, season_base: dict) -> list:
    """
    Returns a list of hex colors (one per row in df). Each crop gets a shade of
    its season's hue following the fixed _SHADE_PATTERN, so crops are visually
    distinct but clearly grouped by season.
    """
    season_shades: dict = {}
    for season in df["season"].unique():
        n = len(df[df["season"] == season])
        base = season_base.get(season, "#888888")
        hue, _, sat = _hex_to_hls(base)
        shades = [
            _hls_to_hex(hue, lightness, min(sat * s_scale, 1.0))
            for lightness, s_scale in (_SHADE_PATTERN * ((n // len(_SHADE_PATTERN)) + 1))[:n]
        ]
        season_shades[season] = {"shades": shades, "idx": 0}

    colors = []
    for _, row in df.iterrows():
        info = season_shades[row["season"]]
        colors.append(info["shades"][info["idx"]])
        info["idx"] += 1
    return colors


# ── Fertilizer calculation ────────────────────────────────────────────────────
def get_fertilizer_totals(crops_df: pd.DataFrame) -> dict:
    """
    Receives the farmer's crops DataFrame (columns: crop_name, acres).
    Returns a dict with keys:
        fertilizer_used_lbs (float)  |  fertilizer_cost_usd (float)
    """
    # Use N_INTENSITY_LBS_PER_ACRE constant + current urea price from cache
    _cache = load_cache()
    current_price_per_mt = (_cache or {}).get("signal", {}).get("currentPrice", 400)
    # Convert $/mt → $/lb of urea → $/lb of N (urea is 46% N)
    cost_per_lb_n = (current_price_per_mt / LBS_PER_MT) / UREA_N_CONTENT
    default_n_lbs = 100  # lbs N/acre for unknown crops
    used, cost = 0.0, 0.0
    for _, row in crops_df.iterrows():
        n_lbs_per_acre = N_INTENSITY_LBS_PER_ACRE.get(row["crop_name"], default_n_lbs)
        used += n_lbs_per_acre * row["acres"]
        cost += n_lbs_per_acre * row["acres"] * cost_per_lb_n
    return {"fertilizer_used_lbs": used, "fertilizer_cost_usd": cost}


# ── State fertilizer exposure builder ────────────────────────────────────────
def build_state_df(cache: dict) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per state containing:
      urea_mt         — total urea metric tons needed annually
      current_cost_m  — fertilizer bill at current price ($M)
      forecast_cost_m — fertilizer bill at t2 forecast price ($M)
      impact_m        — cost change vs. today ($M, positive = more expensive)
      elevation       — column height in metres for PyDeck (normalised, max 500 km)
      color           — RGBA list for PyDeck heat gradient
      tooltip         — HTML tooltip string
    """
    sig           = cache.get("signal", {})
    current_price = sig.get("currentPrice", 0) or 1  # guard against zero division
    forecast_t2   = sig.get("forecast_t2", current_price)
    price_chg_pct = (forecast_t2 - current_price) / current_price * 100

    rows = []
    for state, (lat, lng, corn_k, wheat_k, soy_k) in _STATE_AG.items():
        n_lbs     = (corn_k  * 1000 * N_INTENSITY_LBS_PER_ACRE["Corn"]
                     + wheat_k * 1000 * N_INTENSITY_LBS_PER_ACRE["Wheat"]
                     + soy_k   * 1000 * N_INTENSITY_LBS_PER_ACRE["Soybeans"])
        urea_mt   = n_lbs / (UREA_N_CONTENT * LBS_PER_MT)
        cur_cost  = urea_mt * current_price / 1e6
        fc_cost   = urea_mt * forecast_t2   / 1e6
        impact    = fc_cost - cur_cost
        rows.append({
            "state":          state,
            "lat":            lat,
            "lng":            lng,
            "corn_acres":     corn_k * 1000,
            "wheat_acres":    wheat_k * 1000,
            "soy_acres":      soy_k  * 1000,
            "urea_mt":        urea_mt,
            "current_cost_m": cur_cost,
            "forecast_cost_m":fc_cost,
            "impact_m":       impact,
        })

    df = pd.DataFrame(rows)

    # Elevation: normalise urea_mt so the tallest state = 500 km
    max_mt        = df["urea_mt"].max()
    df["elevation"] = (df["urea_mt"] / max_mt) * 500_000

    # Heat color: green (low exposure) → yellow → red (high exposure)
    intensity        = df["urea_mt"] / max_mt          # 0-1
    df["color_r"]    = (intensity * 255).astype(int)
    df["color_g"]    = (255 - intensity * 200).clip(0, 255).astype(int)
    df["color_b"]    = 30
    df["color_a"]    = 210

    sign = "▲" if price_chg_pct >= 0 else "▼"
    df["tooltip"] = df.apply(lambda r: (
        f"<b>{r['state']}</b><br>"
        f"Urea needed: {r['urea_mt']/1000:,.0f}K mt/yr<br>"
        f"Current bill: ${r['current_cost_m']:.1f}M<br>"
        f"60-day forecast: ${r['forecast_cost_m']:.1f}M<br>"
        f"Impact: {sign}${abs(r['impact_m']):.1f}M ({price_chg_pct:+.1f}%)"
    ), axis=1)

    return df.sort_values("urea_mt", ascending=False)


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gas Forecast — Dashboard", page_icon="🌾", layout="wide", initial_sidebar_state="collapsed")

# Load banner background image
_banner_img_b64 = get_image_base64("image.png")
_banner_bg_css = (
    f"linear-gradient(135deg,rgba(14,60,25,0.80) 0%,rgba(16,80,36,0.72) 100%), url({_banner_img_b64})"
    if _banner_img_b64
    else "linear-gradient(135deg,#14532d 0%,#166534 55%,#15803d 100%)"
)

st.markdown("""
<style>
    [data-testid='stSidebarNav'] { display: none; }
    [data-testid='collapsedControl'] { display: none; }
    section[data-testid='stSidebar'] { display: none; }
    header[data-testid='stHeader'] { display: none; }
    [data-testid='stToolbar'] { display: none; }
    #MainMenu { display: none; }
    body, .stApp { background-color: #ffffff; }
    h1, h2, h3 { color: #1a5c2a; font-family: system-ui, sans-serif; }

    /* shrink top padding so content starts higher */
    .block-container { padding-top: 0.5rem !important; }

    /* Sticky header row — first direct child of the top-level vertical block */
    section[data-testid="stMain"] > div > div[data-testid="stVerticalBlock"] > div:first-child {
        position: sticky;
        top: 0;
        z-index: 999;
        background: white;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #e8f4e8;
        width: 100vw;
        left: 0;
        margin-left: -1rem;
    }

    /* Keep "Logged in as …" on one line */
    .user-email-bar {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        text-align: right;
        font-size: 0.95rem;
        color: #444;
    }
    
    .user-email-bar img {
        width: 24px;
        height: 24px;
    }

    /* Prevent button labels from wrapping */
    .stButton button, button[data-testid^="stBaseButton"] {
        white-space: nowrap !important;
    }


    .metric-icon {
        width: 48px;
        height: 48px;
        margin: 0 auto 0.8rem;
        display: block;
    }

    .advice-card {
        background: #f9fbf9;
        border-left: 4px solid #1a5c2a;
        border-radius: 0px;
        padding: 2rem 1.8rem;
        margin-top: 1.5rem;
    }

    .risk-medium { color: #e67e22; font-weight: 600; }
    .risk-low    { color: #27ae60; font-weight: 600; }
    .risk-high   { color: #e74c3c; font-weight: 600; }

    /* make tab labels bigger */
    button[data-baseweb="tab"] { font-size: 1.05rem !important; }

    /* ── Banner ── */
    .agri-banner {
        background: linear-gradient(135deg, #14532d 0%, #166534 55%, #15803d 100%);
        padding: 18px 28px;
        margin-bottom: 0.5rem;
        border-radius: 0;
        margin-left: -28px;
        margin-right: -28px;
        width: calc(100% + 56px);
    }
    .agri-banner h1 {
        color: #bbf7d0;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0 0 4px 0;
        letter-spacing: -0.3px;
    }
    .agri-banner p {
        color: #bbf7d0;
        font-size: 0.82rem;
        margin: 0;
        opacity: 0.9;
    }

    /* ── Full-height columns ── */
    [data-testid="stHorizontalBlock"] { align-items: stretch !important; }

    /* ── Center Add button text ── */
    [data-testid="stFormSubmitButton"] button p {
        width: 100%;
        text-align: center !important;
    }

</style>
""", unsafe_allow_html=True)

init_farm_db()
email = st.session_state.user_email
with st.spinner("Loading your farm data..."):
    df    = get_farmer_crops(email)
    cache = load_cache()

# ── Season session state ──────────────────────────────────────────────────────
if "seasons" not in st.session_state:
    st.session_state.seasons = list(DEFAULT_SEASONS)

# ── Top-right user bar ────────────────────────────────────────────────────────
_, col_user, col_signout = st.columns([5, 3, 1])
with col_user:
    account_icon_b64 = get_image_base64("account.png")
    icon_html = f'<img src="{account_icon_b64}" alt="account" style="width:20px; height:20px; margin-right:6px;"/>' if account_icon_b64 else ""
    st.markdown(f'<div class="user-email-bar">{icon_html}<span>Logged in as <strong>{email}</strong></span></div>', unsafe_allow_html=True)
with col_signout:
    if st.button("Sign Out", width='stretch'):
        _tok = st.session_state.pop("_session_token", None)
        if _tok:
            try:
                sqlite3.connect(DB_PATH).execute(
                    "DELETE FROM session_tokens WHERE token = ?", (_tok,)
                ).connection.commit()
            except Exception:
                pass
        st.query_params.clear()
        st.session_state.logged_in  = False
        st.session_state.user_email = ""
        st.switch_page("pages/login.py")

# ── Banner ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="agri-banner">
  <h1>🌾 foreGASt</h1>
  <p>Fertilizer intelligence for American farmers. Real prices, real risk, real decisions.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_overview, tab_fertilizer, tab_map = st.tabs(["Overview", "Fertilizer Costs & Risk Assessment", "🌍 Regional Price Map"])


def _build_sparkline_svg(values: list, width: int = 200, height: int = 48) -> str:
    """Returns an inline SVG polyline from a list of floats. Empty string if insufficient data."""
    if not values or len(values) < 2:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    step = width / (len(values) - 1)
    pts = " ".join(
        f"{i * step:.1f},{height - ((v - mn) / rng) * (height - 6) - 3:.1f}"
        for i, v in enumerate(values)
    )
    fill_pts = f"0,{height} " + pts + f" {width},{height}"
    last_x = (len(values) - 1) * step
    last_y = height - ((values[-1] - mn) / rng) * (height - 6) - 3
    return (
        f'<svg viewBox="0 0 {width} {height}" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" style="width:100%;height:{height}px;">'
        f'<polygon points="{fill_pts}" fill="rgba(134,239,172,0.15)"/>'
        f'<polyline points="{pts}" stroke="#86efac" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3" fill="#4ade80"/>'
        f'</svg>'
    )


def _build_cost_bars_html(df: pd.DataFrame, fert_totals: dict) -> str:
    """Returns HTML string of per-crop horizontal cost bars for the green financial panel."""
    if df.empty:
        return '<div style="color:#86efac;font-size:0.75rem;opacity:0.7;">Add crops to see breakdown.</div>'

    _cache = load_cache()
    cur_p = (_cache or {}).get("signal", {}).get("currentPrice", 400) or 400
    cost_per_lb_n = (cur_p / LBS_PER_MT) / UREA_N_CONTENT

    from collections import defaultdict
    agg: dict = defaultdict(float)
    for _, row in df.iterrows():
        n_lbs = N_INTENSITY_LBS_PER_ACRE.get(row["crop_name"], 100) * row["acres"]
        agg[row["crop_name"]] += n_lbs * cost_per_lb_n

    total = sum(agg.values()) or 1
    bar_colors = ["#4ade80", "#34d399", "#22c55e", "#16a34a", "#15803d"]
    crop_emojis = {"Corn": "🌽", "Wheat": "🌾", "Soybeans": "🫘",
                   "Cotton": "🪴", "Sorghum": "🌿", "Hay": "🌱"}
    html = ""
    for i, (crop, cost) in enumerate(sorted(agg.items(), key=lambda x: -x[1])):
        pct = cost / total * 100
        color = bar_colors[min(i, len(bar_colors) - 1)]
        emoji = crop_emojis.get(crop, "🌾")
        html += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;">'
            f'<span style="font-size:0.65rem;color:#d1fae5;width:72px;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis;">{emoji} {crop}</span>'
            f'<div style="flex:1;height:5px;background:rgba(255,255,255,0.12);'
            f'border-radius:3px;overflow:hidden;">'
            f'<div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:3px;"></div>'
            f'</div>'
            f'<span style="font-size:0.62rem;color:#d1fae5;font-weight:700;'
            f'width:42px;text-align:right;">${cost/1000:.1f}K</span>'
            f'</div>'
        )

    total_val = fert_totals.get("fertilizer_cost_usd", 0)
    html += (
        f'<div style="display:flex;justify-content:space-between;'
        f'border-top:1px dashed rgba(255,255,255,0.2);margin-top:5px;padding-top:5px;">'
        f'<span style="font-size:0.62rem;color:#86efac;">Total</span>'
        f'<span style="font-size:0.68rem;font-weight:700;color:#fff;">${total_val:,.0f}</span>'
        f'</div>'
    )
    return html


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    # ── Shared computed values ────────────────────────────────────────────
    fert          = get_fertilizer_totals(df) if not df.empty else {"fertilizer_used_lbs": 0, "fertilizer_cost_usd": 0}
    total_acres   = float(df["acres"].sum()) if not df.empty else 0.0
    total_cost    = fert["fertilizer_cost_usd"]
    cost_per_acre = (total_cost / total_acres) if total_acres > 0 else 0.0
    n_crops       = int(df["crop_name"].nunique()) if not df.empty else 0

    sig           = (cache or {}).get("signal", {})
    cur_price     = sig.get("currentPrice", 0) or 0
    signal_label  = sig.get("signal", "N/A")
    ng_chg        = sig.get("ng_change_30d", 0) or 0

    urea_hist     = (cache or {}).get("urea_history", None)

    col_left, col_center, col_right = st.columns([1, 1.1, 1.4])

    # ── COLUMN 1: Financial Panel ─────────────────────────────────────────
    with col_left:
        spark_svg = ""
        if urea_hist and urea_hist.get("values"):
            spark_svg = _build_sparkline_svg(urea_hist["values"][-12:])

        if abs(ng_chg) > 0.05:
            direction = "up" if ng_chg > 0 else "down"
            insight_html = (
                f'<div style="background:rgba(255,255,255,0.10);border-radius:8px;'
                f'padding:8px 10px;font-size:0.68rem;color:#d1fae5;line-height:1.5;margin-top:6px;">'
                f'💡 Nat gas {direction} {abs(ng_chg)*100:.0f}% last 30 days — '
                f'watch for urea cost changes in 6–8 weeks.</div>'
            )
        elif df.empty:
            insight_html = (
                '<div style="background:rgba(255,255,255,0.10);border-radius:8px;'
                'padding:8px 10px;font-size:0.68rem;color:#d1fae5;line-height:1.5;margin-top:6px;">'
                '💡 Add crops to see your cost exposure.</div>'
            )
        else:
            insight_html = ""

        cost_bars_html = _build_cost_bars_html(df, fert)

        hero_val   = f"${total_cost:,.0f}" if not df.empty else "—"
        acres_line = (
            f"{total_acres:,.0f} acres · {n_crops} crop{'s' if n_crops != 1 else ''}"
            if not df.empty else "No crops added yet"
        )
        per_acre  = f"${cost_per_acre:.2f}" if not df.empty else "—"
        urea_disp = f"${cur_price:.0f}/mt" if cur_price else "—"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#14532d 0%,#166534 55%,#15803d 100%);
                    border-radius:12px;padding:20px 18px;
                    display:flex;flex-direction:column;gap:10px;">
          <div>
            <div style="font-size:0.58rem;text-transform:uppercase;letter-spacing:1px;
                        color:#86efac;margin-bottom:4px;">Season Fertilizer Exposure</div>
            <div style="font-size:2rem;font-weight:800;color:#fff;line-height:1.1;">{hero_val}</div>
            <div style="font-size:0.68rem;color:#bbf7d0;margin-top:4px;">{acres_line}</div>
          </div>
          {spark_svg}
          <div style="display:flex;gap:14px;border-top:1px solid rgba(255,255,255,0.12);padding-top:10px;">
            <div>
              <div style="font-size:0.52rem;color:#6ee7b7;text-transform:uppercase;
                          letter-spacing:0.5px;">Per Acre</div>
              <div style="font-size:0.85rem;font-weight:700;color:#fff;margin-top:1px;">{per_acre}</div>
            </div>
            <div>
              <div style="font-size:0.52rem;color:#6ee7b7;text-transform:uppercase;
                          letter-spacing:0.5px;">Urea Price</div>
              <div style="font-size:0.85rem;font-weight:700;color:#fff;margin-top:1px;">{urea_disp}</div>
            </div>
            <div>
              <div style="font-size:0.52rem;color:#6ee7b7;text-transform:uppercase;
                          letter-spacing:0.5px;">60-Day Signal</div>
              <div style="font-size:0.85rem;font-weight:700;color:#fff;margin-top:1px;">{signal_label}</div>
            </div>
          </div>
          <div style="border-top:1px solid rgba(255,255,255,0.12);padding-top:8px;">
            <div style="font-size:0.55rem;color:#86efac;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:6px;">Cost by Crop</div>
            {cost_bars_html}
          </div>
          {insight_html}
        </div>
        """, unsafe_allow_html=True)

    # ── COLUMN 2: Crop Distribution ───────────────────────────────────────
    with col_center:
        st.markdown(
            '<div style="font-size:0.62rem;color:#4b7c59;font-weight:600;'
            'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">'
            'Crop Distribution</div>',
            unsafe_allow_html=True,
        )

        if df.empty:
            st.markdown("""
            <div style="background:#f0fdf4;border-radius:12px;height:260px;
                        display:flex;align-items:center;justify-content:center;
                        border:2px dashed #86efac;text-align:center;padding:20px;">
              <div>
                <div style="font-size:2rem;margin-bottom:8px;">🌾</div>
                <div style="font-size:0.85rem;color:#4b7c59;font-weight:600;">No crops yet</div>
                <div style="font-size:0.72rem;color:#9ca3af;margin-top:4px;">
                  Add your first crop on the right →</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            # Year filter
            all_years_in_data = sorted(df["year"].dropna().astype(int).unique().tolist(), reverse=True)
            current_year = 2026
            default_year = current_year if current_year in all_years_in_data else (all_years_in_data[0] if all_years_in_data else current_year)
            if "pie_year_filter" not in st.session_state:
                st.session_state["pie_year_filter"] = default_year
            if st.session_state["pie_year_filter"] not in all_years_in_data and all_years_in_data:
                st.session_state["pie_year_filter"] = default_year
            selected_year = st.selectbox(
                "Year",
                options=all_years_in_data,
                index=all_years_in_data.index(st.session_state["pie_year_filter"]) if st.session_state["pie_year_filter"] in all_years_in_data else 0,
                key="pie_year_filter",
            )

            # Season filter
            all_seasons_in_data   = sorted(df["season"].unique().tolist())
            all_seasons_available = sorted(set(all_seasons_in_data + st.session_state.seasons))
            if "_pie_select_all_saved" in st.session_state:
                st.session_state["pie_select_all"] = st.session_state.pop("_pie_select_all_saved")
            if "_pie_filter_saved" in st.session_state:
                saved = st.session_state.pop("_pie_filter_saved")
                if saved is not None:
                    st.session_state["pie_season_filter"] = saved
            if "pie_select_all" not in st.session_state:
                st.session_state["pie_select_all"] = True
            select_all = st.checkbox("Select all seasons", key="pie_select_all")
            selected_seasons = (
                all_seasons_available if select_all else
                st.multiselect("Filter by season", options=all_seasons_available,
                               default=all_seasons_in_data, key="pie_season_filter")
            )

            filtered_df = df[
                (df["season"].isin(selected_seasons)) &
                (df["year"].astype(int) == int(selected_year))
            ].copy() if selected_seasons else df.iloc[0:0].copy()

            season_base = get_season_base_colors(all_seasons_available)

            if filtered_df.empty:
                st.info("No crops match the selected year and seasons.")
            else:
                crop_colors = get_crop_colors_for_df(filtered_df, season_base)
                dup_crops = filtered_df["crop_name"].duplicated(keep=False)
                filtered_df["display_label"] = filtered_df.apply(
                    lambda r: f"{r['crop_name']} ({r['season']})" if dup_crops[r.name] else r["crop_name"],
                    axis=1,
                )
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=filtered_df["display_label"],
                    values=filtered_df["acres"],
                    marker=dict(colors=crop_colors, line=dict(color="white", width=1.5)),
                    textposition="inside",
                    textinfo="percent+label",
                    hole=0.35,
                    hovertemplate="<b>%{label}</b><br>%{value:,.0f} acres (%{percent})<extra></extra>",
                    showlegend=True,
                ))
                for season, base_color in season_base.items():
                    if season in filtered_df["season"].values:
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None], mode="markers",
                            marker=dict(symbol="square", size=10, color=base_color),
                            name=season, showlegend=True,
                        ))
                fig.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.28,
                                xanchor="center", x=0.5, font=dict(size=11)),
                    margin=dict(t=0, b=55, l=10, r=10),
                    height=270,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#14532d"),
                    xaxis=dict(visible=False, showgrid=False),
                    yaxis=dict(visible=False, showgrid=False),
                )
                st.markdown('<div style="margin-top:-12px;">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Insight cards
                _c2p   = (cache or {}).get("signal", {}).get("currentPrice", 400) or 400
                _cplbn = (_c2p / LBS_PER_MT) / UREA_N_CONTENT
                crop_costs = {}
                for crop_name in df["crop_name"].unique():
                    total_n = df[df["crop_name"] == crop_name]["acres"].sum() * N_INTENSITY_LBS_PER_ACRE.get(crop_name, 100)
                    crop_costs[crop_name] = total_n * _cplbn

                top_crop = max(crop_costs, key=crop_costs.get) if crop_costs else "—"
                top_cost = crop_costs.get(top_crop, 0)
                top_pct  = top_cost / sum(crop_costs.values()) * 100 if crop_costs else 0
                hi_n_crop = max(df["crop_name"].unique(),
                                key=lambda c: N_INTENSITY_LBS_PER_ACRE.get(c, 0),
                                default="—")
                hi_n_val = N_INTENSITY_LBS_PER_ACRE.get(hi_n_crop, 0)

                st.markdown(f"""
                <div style="display:flex;gap:8px;margin-top:-30px;margin-bottom:4px;align-items:stretch;">
                  <div style="flex:1;background:#fff;border:1.5px solid #d1fae5;border-radius:10px;
                              padding:10px 12px;">
                    <div style="font-size:0.55rem;color:#6b7280;text-transform:uppercase;
                                letter-spacing:0.4px;margin-bottom:3px;">Largest Cost Driver</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#14532d;">{top_crop}</div>
                    <div style="font-size:0.6rem;color:#6b7280;margin-top:2px;">
                      ${top_cost:,.0f} · {top_pct:.0f}% of spend</div>
                  </div>
                  <div style="flex:1;background:#fff;border:1.5px solid #d1fae5;border-radius:10px;
                              padding:10px 12px;display:flex;align-items:center;">
                    <div>
                      <div style="font-size:0.55rem;color:#6b7280;text-transform:uppercase;
                                  letter-spacing:0.4px;margin-bottom:3px;">Highest N Intensity</div>
                      <div style="font-size:0.95rem;font-weight:700;color:#14532d;">{hi_n_val} lbs/ac</div>
                      <div style="font-size:0.6rem;color:#6b7280;margin-top:2px;">{hi_n_crop} — most N-heavy</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── COLUMN 3: Crop Management ─────────────────────────────────────────
    with col_right:
        st.markdown(
            '<div style="font-size:0.62rem;color:#4b7c59;font-weight:600;'
            'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">'
            'Your Crops</div>',
            unsafe_allow_html=True,
        )

        # Crop list
        if not df.empty:
            season_base_r = get_season_base_colors(st.session_state.seasons)
            crop_colors_r = get_crop_colors_for_df(df, season_base_r)
            for (_, row), color in zip(df.iterrows(), crop_colors_r):
                r1, r2 = st.columns([4, 1])
                r1.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
                    f'<div style="width:9px;height:9px;border-radius:50%;background:{color};'
                    f'flex-shrink:0;"></div>'
                    f'<div>'
                    f'<div style="font-size:0.75rem;font-weight:600;color:#14532d;">{row["crop_name"]}</div>'
                    f'<div style="font-size:0.62rem;color:#6b7280;">'
                    f'{row["season"]} · {row["acres"]:,.0f} ac · '
                    f'${N_INTENSITY_LBS_PER_ACRE.get(row["crop_name"],100)*row["acres"]*(cur_price/LBS_PER_MT/UREA_N_CONTENT if cur_price else 0):,.0f}'
                    f'</div></div></div>',
                    unsafe_allow_html=True,
                )
                if r2.button("✕", key=f"del_{row['id']}", use_container_width=True):
                    delete_crop(row["id"])
                    st.rerun()
        else:
            st.markdown(
                '<div style="font-size:0.8rem;color:#9ca3af;font-style:italic;'
                'text-align:center;padding:16px 0 8px 0;">Your farm is empty.<br>'
                'Add your first crop below.</div>',
                unsafe_allow_html=True,
            )

        # Add crop form
        with st.expander("＋ Add a Crop", expanded=df.empty):
            COMMON_CROPS = [
                "Alfalfa", "Barley", "Canola", "Cotton", "Corn", "Hay",
                "Oats", "Rice", "Sorghum", "Soybeans", "Sunflowers", "Wheat",
            ]
            if "custom_crops" not in st.session_state:
                st.session_state.custom_crops = []
            if "_next_crop" in st.session_state:
                st.session_state["crop_selectbox"] = st.session_state.pop("_next_crop")
            if "_next_season" in st.session_state:
                st.session_state["season_selectbox"] = st.session_state.pop("_next_season")

            all_crops = COMMON_CROPS + st.session_state.custom_crops + ["Add new crop…"]
            col_a, col_c = st.columns([1, 1])
            crop_sel = col_a.selectbox("Crop", options=all_crops, key="crop_selectbox")
            if crop_sel == "Add new crop…":
                new_crop_text = col_a.text_input("New crop name", placeholder="e.g. Millet", key="new_crop_text")
                if col_a.button("Add to list", key="add_crop_btn"):
                    nc = new_crop_text.strip().title()
                    if nc and nc not in COMMON_CROPS and nc not in st.session_state.custom_crops:
                        st.session_state.custom_crops.append(nc)
                    if nc:
                        st.session_state["_next_crop"] = nc
                        st.rerun()

            season_options = st.session_state.seasons + ["Create new…"]
            season_sel = col_c.selectbox("Season", options=season_options, key="season_selectbox")
            if season_sel == "Create new…":
                new_season_text = col_c.text_input("New season name", placeholder="e.g. Summer", key="new_season_text")
                if col_c.button("Add season", key="add_season_btn"):
                    ns = new_season_text.strip().title()
                    if ns and ns not in st.session_state.seasons:
                        st.session_state.seasons.append(ns)
                    if ns:
                        st.session_state["_next_season"] = ns
                        st.rerun()

            with st.form("add_crop_form", clear_on_submit=True):
                col_b, col_yr, col_d = st.columns([3, 3, 3], vertical_alignment="bottom")
                acres_input = col_b.text_input("Acres", placeholder="e.g. 320")
                year_sel = col_yr.number_input("Year", min_value=2000, max_value=2100, value=2025, step=1, key="year_input")
                submitted = col_d.form_submit_button("Add", use_container_width=True)
                if submitted:
                    crop_val   = st.session_state.get("crop_selectbox", "")
                    season_val = st.session_state.get("season_selectbox", "")
                    year_val   = st.session_state.get("year_input", 2025)
                    if not crop_val or crop_val == "Add new crop…":
                        st.warning("Please select (or add) a crop first.")
                        st.stop()
                    if not season_val or season_val == "Create new…":
                        st.warning("Please select (or create) a season first.")
                        st.stop()
                    try:
                        acres_val = float(acres_input.strip().replace(",", ""))
                        if acres_val <= 0:
                            raise ValueError
                    except ValueError:
                        st.warning("Please enter a valid number of acres.")
                        st.stop()
                    add_crop(email, crop_val, acres_val, season_val, int(year_val))
                    st.success(f"Added {crop_val} — {season_val} {int(year_val)} ({acres_val:,.0f} acres).")
                    st.session_state["_pie_select_all_saved"] = st.session_state.get("pie_select_all", True)
                    st.session_state["_pie_filter_saved"] = st.session_state.get("pie_season_filter", None)
                    st.rerun()

        # Signal pill
        _sig_colors = {
            "BUY_NOW": "#ef4444", "CONSIDER_BUYING": "#f59e0b",
            "WAIT": "#22c55e", "NEUTRAL": "#6b7280",
        }
        _dot_color = _sig_colors.get(signal_label, "#6b7280")
        _rationale = sig.get("rationale", "Run pipeline for signal.")
        _rationale_short = _rationale[:70] + ("…" if len(_rationale) > 70 else "")
        st.markdown(f"""
        <div style="background:#fff;border:1.5px solid #d1fae5;border-radius:8px;
                    padding:8px 12px;display:flex;align-items:flex-start;gap:8px;margin-top:8px;">
          <div style="width:9px;height:9px;border-radius:50%;background:{_dot_color};
                      flex-shrink:0;margin-top:3px;"></div>
          <div style="font-size:0.68rem;color:#14532d;line-height:1.5;">
            <strong>{signal_label}</strong> — {_rationale_short}<br>
            <span style="color:#4b7c59;font-size:0.62rem;">
              See "Fertilizer Costs &amp; Risk Assessment" tab for full analysis.</span>
          </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — FERTILIZER COSTS & RISK ASSESSMENT
# ════════════════════════════════════════════════════════════════════════════
with tab_fertilizer:

    if cache is None:
        st.warning(
            "No forecast data found. "
            "Run `python backend/run_pipeline.py` first to generate `data/processed/cache.json`."
        )
    else:
        sig  = cache["signal"]
        mc   = cache["monte_carlo"]
        fc   = cache["forecast"]
        meta = cache["model_metadata"]

        SIGNAL_COLORS = {
            "BUY_NOW":         "#ef4444",
            "CONSIDER_BUYING": "#f59e0b",
            "WAIT":            "#22c55e",
            "NEUTRAL":         "#6b7280",
        }
        color = SIGNAL_COLORS.get(sig.get("signal", "NEUTRAL"), "#6b7280")

        # ── SECTION 1: BUY SIGNAL ─────────────────────────────────────────────
        st.subheader("📡 Buy Signal")
        with st.expander("ℹ️ How to read this"):
            st.markdown("""
**Why natural gas?** Urea (the most common nitrogen fertilizer) is manufactured from natural gas.
When nat gas prices rise, fertilizer factories pay more, and that cost passes to farmers 4–8 weeks later.
This model exploits that lag to give you advance warning.

**The four signals:**
| Signal | Meaning |
|---|---|
| 🔴 **BUY NOW** | Prices forecast to rise >8% in 60 days with high confidence — lock in supply today |
| 🟡 **CONSIDER BUYING** | Moderate upward pressure — partial pre-purchase recommended |
| 🟢 **WAIT** | Prices forecast to fall — buying now would cost more than waiting |
| ⚫ **NEUTRAL** | No strong signal — forecast is flat and probability is evenly split |

**Confidence** is the Monte Carlo probability that price will be higher in 60 days.
**Best month** is whichever of the 3 forecast months has the lowest median simulated price.
            """)

        cur_price = sig.get("currentPrice", 0) or 1
        fc_t2     = sig.get("forecast_t2", cur_price)
        pct_chg   = (fc_t2 - cur_price) / cur_price * 100
        prob_r    = sig.get("prob_rising", 0)
        ng_cur    = sig.get("ng_current", 0)
        ng_chg    = sig.get("ng_change_30d", 0)

        pct_color  = "#22c55e" if pct_chg >= 0 else "#ef4444"
        pct_arrow  = "▲" if pct_chg >= 0 else "▼"
        ng_color   = "#22c55e" if ng_chg >= 0 else "#ef4444"
        ng_arrow   = "▲" if ng_chg >= 0 else "▼"

        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-bottom:8px;">
          <div style="flex:1;border:1.5px solid #d1fae5;border-radius:10px;padding:14px 16px;
                      background:#fff;box-shadow:inset 0 0 0 3px rgba(34,197,94,0.08);">
            <div style="font-size:0.65rem;color:#6b7280;margin-bottom:6px;">Current Urea Price</div>
            <div style="font-size:1.4rem;font-weight:700;color:#14532d;">${cur_price:.0f}/mt</div>
          </div>
          <div style="flex:1;border:1.5px solid #d1fae5;border-radius:10px;padding:14px 16px;
                      background:#fff;box-shadow:inset 0 0 0 3px rgba(34,197,94,0.08);">
            <div style="font-size:0.65rem;color:#6b7280;margin-bottom:6px;">60-Day Forecast (t2)</div>
            <div style="font-size:1.4rem;font-weight:700;color:#14532d;">${fc_t2:.0f}/mt</div>
            <div style="font-size:0.75rem;color:{pct_color};margin-top:4px;">{pct_arrow} {pct_chg:+.1f}%</div>
          </div>
          <div style="flex:1;border:1.5px solid #d1fae5;border-radius:10px;padding:14px 16px;
                      background:#fff;box-shadow:inset 0 0 0 3px rgba(34,197,94,0.08);">
            <div style="font-size:0.65rem;color:#6b7280;margin-bottom:6px;">Prob. Rising (60d)</div>
            <div style="font-size:1.4rem;font-weight:700;color:#14532d;">{prob_r*100:.0f}%</div>
          </div>
          <div style="flex:1;border:1.5px solid #d1fae5;border-radius:10px;padding:14px 16px;
                      background:#fff;box-shadow:inset 0 0 0 3px rgba(34,197,94,0.08);">
            <div style="font-size:0.65rem;color:#6b7280;margin-bottom:6px;">Nat Gas Spot</div>
            <div style="font-size:1.4rem;font-weight:700;color:#14532d;">${ng_cur:.2f}/MMBtu</div>
            <div style="font-size:0.75rem;color:{ng_color};margin-top:4px;">{ng_arrow} {ng_chg*100:+.1f}% (30d)</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{color}22; border-left:4px solid {color}; padding:16px;
                    border-radius:6px; margin:12px 0 4px 0;">
            <span style="font-size:1.35rem; font-weight:bold; color:{color};">
                {sig.get('recommendation', 'N/A')}
            </span>
            &nbsp;&nbsp;
            <span style="color:#666; font-size:0.88rem;">
                Confidence: {sig.get('confidence', 0)*100:.0f}%
                &nbsp;|&nbsp; {sig.get('key_driver', 'N/A')}
                &nbsp;|&nbsp; Best month: {sig.get('bestMonth', 'N/A')} (${sig.get('bestPrice', 0):.0f}/mt)
            </span>
            <br>
            <span style="color:#444; font-size:0.95rem; margin-top:8px; display:block;">
                {sig.get('rationale', '')}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── SECTION 2: PRICE FORECAST CHART ───────────────────────────────────
        st.subheader("📈 Urea Price History + XGBoost Forecast")
        with st.expander("ℹ️ How to read this"):
            st.markdown("""
**Blue line** — actual urea spot prices ($/metric ton) from Jan 2018 to the most recent data month.

**Red dashed line** — the XGBoost machine-learning model's point forecast for the next 3 months.
XGBoost learns patterns between natural gas prices, storage levels, momentum, and seasonality to
predict where urea prices are heading.

**Red shaded band** — the 80% Monte Carlo confidence interval. The model ran 10,000 simulated futures
and this band captures the middle 80% of outcomes. If the band is wide, uncertainty is high.
If it's narrow, the model is more confident.

**Nat Gas overlay (toggle)** — shows US natural gas spot price on the right axis (green dotted line), **shifted forward 5 months** to align with its actual effect on urea prices. Urea is manufactured from natural gas; a price move in nat gas takes ~4–6 months to pass through to fertilizer markets. The shift makes the correlation visible: nat gas spikes should line up with urea spikes on the chart.
            """)

        show_ng = st.toggle("Overlay Natural Gas prices (secondary axis)", value=False)

        def _labels_to_dates(labels, shift_months=0):
            """Convert 'Mon YYYY' strings to ISO date strings for proper Plotly time axis."""
            from datetime import datetime
            from dateutil.relativedelta import relativedelta
            out = []
            for lbl in labels:
                try:
                    dt = datetime.strptime(lbl, "%b %Y")
                    if shift_months:
                        dt = dt + relativedelta(months=shift_months)
                    out.append(dt.strftime("%Y-%m-01"))
                except ValueError:
                    out.append(lbl)
            return out

        hist_x  = _labels_to_dates(cache["urea_history"]["labels"])
        hist_y  = cache["urea_history"]["values"]

        bridge_x    = [hist_x[-1]] + _labels_to_dates(fc["labels"])
        bridge_mean = [hist_y[-1]] + fc["mean"]
        bridge_high = [hist_y[-1]] + fc["high"]
        bridge_low  = [hist_y[-1]] + fc["low"]

        fig_fc = go.Figure()

        fig_fc.add_trace(go.Scatter(
            x=bridge_x + bridge_x[::-1],
            y=bridge_high + bridge_low[::-1],
            fill="toself",
            fillcolor="rgba(239,68,68,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% MC Band",
            hoverinfo="skip",
        ))
        fig_fc.add_trace(go.Scatter(
            x=hist_x, y=hist_y,
            name="Urea — Historical",
            line=dict(color="#2563eb", width=2),
            hovertemplate="$%{y:.0f}/mt<extra>Historical</extra>",
        ))
        fig_fc.add_trace(go.Scatter(
            x=bridge_x, y=bridge_mean,
            name="XGBoost Forecast",
            line=dict(color="#ef4444", width=2.5, dash="dash"),
            hovertemplate="$%{y:.0f}/mt<extra>Forecast (median)</extra>",
        ))
        fig_fc.add_trace(go.Scatter(
            x=_labels_to_dates(fc["labels"]), y=fc["mean"],
            mode="markers",
            marker=dict(color="#ef4444", size=9),
            showlegend=False,
            hoverinfo="skip",
        ))

        if show_ng:
            fig_fc.add_trace(go.Scatter(
                x=_labels_to_dates(cache["natgas_history"]["labels"]),
                y=cache["natgas_history"]["values"],
                name="Nat Gas ($/MMBtu)",
                line=dict(color="#16a34a", width=1.5, dash="dot"),
                yaxis="y2",
                hovertemplate="$%{y:.2f}/MMBtu<extra>Nat Gas</extra>",
            ))
            fig_fc.update_layout(yaxis2=dict(
                title="Nat Gas ($/MMBtu)",
                overlaying="y", side="right",
                showgrid=False, tickfont=dict(color="#16a34a"),
            ))

        fig_fc.update_layout(
            xaxis_title="Month",
            yaxis_title="Urea Price ($/mt)",
            legend=dict(orientation="h", y=1.08, x=0),
            plot_bgcolor="#f9fbf9",
            paper_bgcolor="white",
            font=dict(color="#333"),
            hovermode="x unified",
            height=430,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_fc, width='stretch')
        st.caption(
            f"As-of: {cache.get('as_of_date', 'N/A')}. "
            "Shaded region = 80% Monte Carlo confidence band (p10–p90, 10,000 simulated paths). "
            f"Generated: {cache.get('generated_at', 'N/A')[:10]}."
        )

        st.divider()

        # ── SECTION 3: MONTE CARLO DISTRIBUTION ───────────────────────────────
        st.subheader("🎲 Monte Carlo Price Distribution — 60-Day Horizon")
        st.markdown("Each bar shows how often a simulated price landed in that range across **10,000 paths**.")
        with st.expander("ℹ️ How to read this"):
            st.markdown("""
**What is Monte Carlo simulation?**
Instead of giving you one forecast number, we run 10,000 "what-if" scenarios by adding realistic
random variation (drawn from the model's historical error distribution) to the XGBoost prediction.
The result is a full picture of the range of possible outcomes, not just a point estimate.

**The three lines:**
| Line | Meaning |
|---|---|
| 🟢 **P10 — Optimistic** | Only 10% of simulations ended up *below* this price. A best-case scenario. |
| 🟡 **P50 — Median** | Half of simulations were above, half below. The most likely single outcome. |
| 🔴 **P90 — Pessimistic** | Only 10% of simulations ended up *above* this price. A worst-case budget number. |

**How to use this as a farmer:** The P90 is your "disaster planning" number — the price you'd pay
if things go badly. Budget fertilizer costs using P90, hope for P10.
            """)

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=cache["sim_t2_distribution"],
            nbinsx=60,
            marker_color="#2563eb",
            opacity=0.75,
            name="Simulated Prices",
        ))
        for label, val, clr in [
            ("P10 (Optimistic)",  mc["p10_t2"], "#16a34a"),
            ("P50 (Median)",      mc["p50_t2"], "#f59e0b"),
            ("P90 (Pessimistic)", mc["p90_t2"], "#ef4444"),
        ]:
            fig_mc.add_vline(
                x=val, line_color=clr, line_width=2, line_dash="dot",
                annotation_text=f" {label}: ${val:.0f}",
                annotation_font_color=clr,
                annotation_position="top right",
            )
        fig_mc.update_layout(
            xaxis_title="Simulated Urea Price ($/mt) at t+2 months",
            yaxis_title="Frequency (paths)",
            plot_bgcolor="#f9fbf9",
            paper_bgcolor="white",
            font=dict(color="#333"),
            height=320,
            showlegend=False,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_mc, width='stretch')

        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-bottom:8px;">
          <div style="flex:1;border:1.5px solid #d1fae5;border-radius:10px;padding:14px 16px;
                      background:#fff;box-shadow:inset 0 0 0 3px rgba(34,197,94,0.08);">
            <div style="font-size:0.65rem;color:#6b7280;margin-bottom:6px;">P10 — Optimistic</div>
            <div style="font-size:1.4rem;font-weight:700;color:#14532d;">${mc['p10_t2']:.0f}/mt</div>
          </div>
          <div style="flex:1;border:1.5px solid #d1fae5;border-radius:10px;padding:14px 16px;
                      background:#fff;box-shadow:inset 0 0 0 3px rgba(34,197,94,0.08);">
            <div style="font-size:0.65rem;color:#6b7280;margin-bottom:6px;">P50 — Median</div>
            <div style="font-size:1.4rem;font-weight:700;color:#14532d;">${mc['p50_t2']:.0f}/mt</div>
          </div>
          <div style="flex:1;border:1.5px solid #d1fae5;border-radius:10px;padding:14px 16px;
                      background:#fff;box-shadow:inset 0 0 0 3px rgba(34,197,94,0.08);">
            <div style="font-size:0.65rem;color:#6b7280;margin-bottom:6px;">P90 — Pessimistic</div>
            <div style="font-size:1.4rem;font-weight:700;color:#14532d;">${mc['p90_t2']:.0f}/mt</div>
          </div>
          <div style="flex:1;border:1.5px solid #d1fae5;border-radius:10px;padding:14px 16px;
                      background:#fff;box-shadow:inset 0 0 0 3px rgba(34,197,94,0.08);">
            <div style="font-size:0.65rem;color:#6b7280;margin-bottom:6px;">Prob. &gt;10% Rise</div>
            <div style="font-size:1.4rem;font-weight:700;color:#14532d;">{mc['prob_10pct_increase_t2']*100:.0f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── SECTION 4: EXPOSURE CALCULATOR ────────────────────────────────────
        st.subheader("🧮 Fertilizer Cost Exposure Calculator")
        st.markdown("Project your actual fertilizer bill at each forecast horizon based on your crop mix and acreage.")
        with st.expander("ℹ️ How to use this"):
            st.markdown("""
**Inputs:**
- **Crop type** — different crops need different amounts of nitrogen (corn is the most intensive at 150 lbs N/acre)
- **Acreage** — pre-filled from your farm profile if you've added crops in the Overview tab
- **Pre-purchased %** — if you've already locked in some supply, slide this up to remove it from the calculation

**The table shows three scenarios for each of the next 3 months:**
- **Forecast Price** — XGBoost's point estimate (median of 10,000 simulations)
- **P10 Low** — your cost if prices land in the optimistic 10th percentile
- **P90 High** — your cost if prices land in the pessimistic 90th percentile

**The math:** Nitrogen needed (lbs) ÷ 46% N content ÷ 2,204.6 lbs/mt = metric tons of urea required.
Your cost = urea needed × forecast price.
            """)

        # Pre-populate acreage from user's farm DB if available
        default_acres = float(df["acres"].sum()) if not df.empty else 500.0

        calc_c1, calc_c2 = st.columns(2)
        with calc_c1:
            calc_crop  = st.selectbox("Crop Type", list(N_INTENSITY_LBS_PER_ACRE.keys()), key="calc_crop")
            calc_acres = st.number_input("Acreage", min_value=1.0, value=default_acres, step=50.0, key="calc_acres")
        with calc_c2:
            pre_pct = st.slider("Already pre-purchased (%)", 0, 100, 0, key="pre_pct")
            n_int   = N_INTENSITY_LBS_PER_ACRE[calc_crop]
            total_urea_mt = calc_acres * n_int / (UREA_N_CONTENT * LBS_PER_MT)
            st.info(
                f"**{calc_crop}** → {n_int} lbs N/acre  \n"
                f"Total N needed: **{calc_acres * n_int:,.0f} lbs**  \n"
                f"Urea needed (46% N): **{total_urea_mt:.1f} mt**"
            )

        remaining_mt  = total_urea_mt * (1 - pre_pct / 100)
        cur           = sig["currentPrice"]
        cost_now      = remaining_mt * cur

        cost_rows = []
        for label, mean, low, high in zip(fc["labels"], fc["mean"], fc["low"], fc["high"]):
            cost_rows.append({
                "Month":                 label,
                "Forecast Price ($/mt)": f"${mean:.0f}",
                "P10 — Low ($/mt)":      f"${low:.0f}",
                "P90 — High ($/mt)":     f"${high:.0f}",
                "Your Cost — Expected":  f"${remaining_mt * mean:,.0f}",
                "Your Cost — Low":       f"${remaining_mt * low:,.0f}",
                "Your Cost — High":      f"${remaining_mt * high:,.0f}",
            })

        st.dataframe(pd.DataFrame(cost_rows), width='stretch', hide_index=True)
        st.caption(
            f"{calc_acres:.0f} ac × {n_int} lbs N/ac ÷ (0.46 × 2204.6) = {total_urea_mt:.1f} mt total.  "
            f"{pre_pct}% pre-purchased → **{remaining_mt:.1f} mt remaining to buy.**"
        )

        st.divider()

        # ── SECTION 4b: PRE-PURCHASE OPTIMIZER (CVaR) ─────────────────────────
        st.subheader("🎯 Pre-Purchase Optimizer")
        st.markdown("Finds the statistically **optimal split** between buying fertilizer now vs. waiting.")
        with st.expander("ℹ️ How to use this"):
            st.markdown("""
**The core question:** Should you buy all your fertilizer today, wait until planting, or split the difference?

**How it works:**
The optimizer runs 10,000 simulated future price paths and finds the exact buy-now fraction that
minimizes your *blended* cost — balancing expected savings against worst-case risk.

**Risk tolerance:**
- **Conservative** — heavily penalizes expensive surprises. Will recommend buying more now to avoid risk.
- **Balanced** — equal weight on expected cost and downside protection.
- **Aggressive** — prioritizes the lowest expected cost, accepting more price uncertainty.

**The math (CVaR):** The algorithm minimizes `(1−α)·Expected Cost + α·CVaR₉₀`, where α is your
risk weight. CVaR₉₀ is the average cost across the worst 10% of simulations — a professional
risk metric used in commodity trading desks.

**Three-strategy comparison:** Buy All Now (no risk, today's price locked in) vs. the Optimal Split
(★ recommended) vs. Wait Entirely (full price risk, possible savings or overrun).

**The curve chart** shows the objective value at every possible buy-now fraction. The amber line is
the mathematical optimum — any other split costs you more in expectation.
            """)

        opt_c1, opt_c2 = st.columns(2)
        with opt_c1:
            months_to_plant = st.slider(
                "Months until planting / purchase deadline",
                min_value=1, max_value=3, value=2, key="months_to_plant",
            )
        with opt_c2:
            risk_label = st.radio(
                "Risk tolerance",
                ["Conservative", "Balanced", "Aggressive"],
                index=1, horizontal=True, key="risk_tol",
            )

        horizon_key = {1: "t1", 2: "t2", 3: "t3"}[months_to_plant]
        p50_h = mc[f"p50_{horizon_key}"]
        std_h = meta[f"residual_std_{horizon_key}"]

        rng      = np.random.default_rng(seed=42)
        mc_paths = rng.normal(loc=p50_h, scale=std_h, size=10_000).clip(50, 2000)

        alpha_map = {"Conservative": 0.75, "Balanced": 0.45, "Aggressive": 0.15}
        alpha     = alpha_map[risk_label]

        f_grid   = np.linspace(0, 1, 201)
        obj_vals = []
        for f in f_grid:
            costs = remaining_mt * (f * cur + (1 - f) * mc_paths)
            obj_vals.append((1 - alpha) * costs.mean() + alpha * np.percentile(costs, 90))

        f_star  = float(f_grid[np.argmin(obj_vals)])
        mt_now  = remaining_mt * f_star
        mt_wait = remaining_mt * (1 - f_star)

        def _cost_stats(paths):
            return paths.mean(), np.percentile(paths, 10), np.percentile(paths, 90)

        c_all_now  = np.full(10_000, remaining_mt * cur)
        c_all_wait = remaining_mt * mc_paths
        c_optimal  = remaining_mt * (f_star * cur + (1 - f_star) * mc_paths)

        exp_now,  p10_now,  p90_now  = _cost_stats(c_all_now)
        exp_wait, p10_wait, p90_wait = _cost_stats(c_all_wait)
        exp_opt,  p10_opt,  p90_opt  = _cost_stats(c_optimal)

        pct_now      = round(f_star * 100)
        pct_wait     = 100 - pct_now
        save_vs_wait = exp_wait - exp_opt
        save_sign    = "save" if save_vs_wait >= 0 else "costs"

        rec_color = "#16a34a" if pct_now >= 60 else ("#f59e0b" if pct_now >= 30 else "#2563eb")
        st.markdown(
            f"<div style='border:2px solid {rec_color};border-radius:12px;"
            f"padding:16px 20px;background:{rec_color}18;margin-bottom:12px;'>"
            f"<span style='font-size:1.05rem;font-weight:700;color:{rec_color};'>"
            f"Optimal split: Buy {pct_now}% now · Wait on {pct_wait}%</span><br>"
            f"<span style='color:#555;font-size:0.85rem;'>"
            f"Lock in <b style='color:#111;'>{mt_now:.1f} mt</b> at today's "
            f"<b style='color:#111;'>${cur:,.0f}/mt</b> — "
            f"defer <b style='color:#111;'>{mt_wait:.1f} mt</b> until closer to planting.  "
            f"Expected to <b style='color:#111;'>{save_sign} ${abs(save_vs_wait):,.0f}</b> "
            f"vs. waiting entirely.</span></div>",
            unsafe_allow_html=True,
        )

        oc1, oc2, oc3 = st.columns(3)
        for col, label, exp, p10, p90, is_opt in [
            (oc1, "Buy All Now",                exp_now,  p10_now,  p90_now,  False),
            (oc2, f"Optimal ({pct_now}% now)",  exp_opt,  p10_opt,  p90_opt,  True),
            (oc3, "Wait Entirely",              exp_wait, p10_wait, p90_wait, False),
        ]:
            if is_opt:
                border = "2px solid #f59e0b"
            else:
                border = "1.5px solid #d1d5db"
            badge = "★ RECOMMENDED" if is_opt else ""
            with col:
                st.markdown(
                    f"<div style='border:{border};border-radius:10px;"
                    f"padding:14px 16px;background:#fff;min-height:175px;'>"
                    f"<div style='font-size:0.9rem;font-weight:700;color:#111;'>{label}</div>"
                    + (f"<div style='font-size:0.68rem;font-weight:700;color:#f59e0b;"
                       f"margin-bottom:6px;'>{badge}</div>" if badge else
                       "<div style='margin-bottom:20px;'></div>")
                    + f"<div style='font-size:1.45rem;font-weight:800;color:#111;'>${exp:,.0f}</div>"
                    f"<div style='font-size:0.73rem;color:#64748b;'>expected total cost</div>"
                    f"<hr style='border:none;border-top:1px solid #e5e7eb;margin:10px 0;'>"
                    f"<div style='font-size:0.76rem;color:#64748b;'>"
                    f"P10  <span style='color:#16a34a;'>${p10:,.0f}</span> · "
                    f"P90  <span style='color:#ef4444;'>${p90:,.0f}</span></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        fig_opt = go.Figure()
        fig_opt.add_trace(go.Scatter(
            x=f_grid * 100, y=obj_vals,
            mode="lines", line=dict(color="#2563eb", width=2),
            name="Objective",
        ))
        fig_opt.add_vline(
            x=f_star * 100, line_color="#f59e0b", line_width=2, line_dash="dash",
            annotation_text=f" Optimal: {pct_now}%",
            annotation_font_color="#f59e0b",
        )
        fig_opt.update_layout(
            xaxis_title="Fraction purchased now (%)",
            yaxis_title=f"({1-alpha:.0%}) · E[cost]  +  ({alpha:.0%}) · CVaR₉₀",
            plot_bgcolor="#f9fbf9", paper_bgcolor="white",
            font=dict(color="#333"), height=260,
            margin=dict(t=20, b=40, l=60, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_opt, width='stretch')
        st.caption(
            f"Objective minimised at **{pct_now}%** · "
            f"Risk tolerance: {risk_label} (α = {alpha}) · "
            f"Horizon: {months_to_plant}-month MC forecast (residual σ = ${std_h:.0f}/mt)"
        )

        st.divider()

        # ── SECTION 5: TIMING DECISION CARDS ──────────────────────────────────
        st.subheader("⚖️ Timing Decision: Buy Now vs. Wait")
        st.markdown(
            f"Side-by-side cost comparison for your **remaining {remaining_mt:.1f} mt** to purchase."
        )
        with st.expander("ℹ️ How to read this"):
            st.markdown("""
**Three cards, three choices:**
- **Buy Now** — you pay today's spot price, no uncertainty. This is your baseline.
- **Wait 30 Days** — you defer to next month. The price range shown is the Monte Carlo P10–P90 band.
- **Wait 60 Days** — defer two months. Higher uncertainty (wider band), but more time for prices to move.

**Badge colors:**
- 🔵 **BASELINE** — the reference point (buy now)
- 🟢 **SAVE X%** — the model forecasts you'd pay less by waiting
- 🟡 **X% CHANCE HIGHER** — moderate risk the price will rise if you wait
- 🔴 **X% CHANCE HIGHER** — high probability waiting costs you more

**"Prob cheaper than now"** — the percentage of Monte Carlo simulations where the future price
ended up *below* today's price. Above 50% means waiting is more likely to save money.

**Your cost range** — P10 (best case) to P90 (worst case) applied to your urea quantity.
            """)

        def _prob_rising(p50, std):
            if std <= 0:
                return 1.0 if p50 > cur else 0.0
            return float(1 - _norm.cdf(cur, loc=p50, scale=std))

        scenarios = [
            {
                "label":    "Buy Now",
                "sublabel": "Lock in today's spot price",
                "p10": cur, "p50": cur, "p90": cur,
                "prob_rising": None,
            },
            {
                "label":    "Wait 30 Days",
                "sublabel": fc["labels"][0],
                "p10": mc["p10_t1"], "p50": mc["p50_t1"], "p90": mc["p90_t1"],
                "prob_rising": _prob_rising(mc["p50_t1"], meta["residual_std_t1"]),
            },
            {
                "label":    "Wait 60 Days",
                "sublabel": fc["labels"][1],
                "p10": mc["p10_t2"], "p50": mc["p50_t2"], "p90": mc["p90_t2"],
                "prob_rising": mc["prob_rising_t2"],
            },
        ]

        sc1, sc2, sc3 = st.columns(3)
        for col, s in zip([sc1, sc2, sc3], scenarios):
            p10, p50, p90 = s["p10"], s["p50"], s["p90"]
            c_p50         = remaining_mt * p50
            c_p10         = remaining_mt * p10
            c_p90         = remaining_mt * p90
            delta         = c_p50 - cost_now
            delta_pct     = (delta / cost_now * 100) if cost_now > 0 else 0
            is_baseline   = s["prob_rising"] is None
            prob_save_pct = 0.0 if is_baseline else (1 - s["prob_rising"]) * 100

            price_str      = f"${p50:,.0f} /mt"
            range_str      = "Fixed — no price risk" if is_baseline else f"P10  ${p10:,.0f}  –  P90  ${p90:,.0f} /mt"
            cost_str       = f"${c_p50:,.0f}"
            cost_range_str = f"${c_p10:,.0f} – ${c_p90:,.0f}"
            delta_str      = "No price risk — cost locked in" if is_baseline else f"vs Buy Now:  {delta:+,.0f}  ({delta_pct:+.1f}%)"
            prob_str       = "&nbsp;" if is_baseline else f"Prob cheaper than now:  {prob_save_pct:.0f}%"

            if is_baseline:
                border = "#2563eb"
                bg = "rgba(37,99,235,0.07)"
                b_col = "#2563eb"
                badge = "BASELINE"
            elif p50 < cur:
                border = "#16a34a"
                bg = "rgba(22,163,74,0.07)"
                b_col = "#16a34a"
                badge  = f"SAVE {abs(delta_pct):.1f}%"
            else:
                pct_up = s["prob_rising"] * 100
                if pct_up < 70:
                    border = "#f59e0b"
                    bg = "rgba(245,158,11,0.07)"
                    b_col = "#f59e0b"
                else:
                    border = "#ef4444"
                    bg = "rgba(239,68,68,0.07)"
                    b_col = "#ef4444"
                badge = f"{pct_up:.0f}% CHANCE HIGHER"

            delta_col = "#64748b" if is_baseline else ("#16a34a" if delta <= 0 else "#ef4444")
            prob_col  = "#16a34a" if prob_save_pct >= 50 else "#f59e0b"

            with col:
                st.markdown(
                    f"<div style='border:2px solid {border};border-radius:12px;"
                    f"padding:18px 16px;background:{bg};min-height:300px;"
                    f"display:flex;flex-direction:column;'>"
                    f"<div style='font-size:1.0rem;font-weight:700;color:#111;'>{s['label']}</div>"
                    f"<div style='font-size:0.75rem;color:#64748b;margin-bottom:6px;'>{s['sublabel']}</div>"
                    f"<span style='background:{b_col}22;color:{b_col};font-size:0.68rem;"
                    f"font-weight:700;padding:3px 9px;border-radius:20px;'>{badge}</span>"
                    f"<div style='margin-top:14px;font-size:1.9rem;font-weight:800;"
                    f"color:#111;line-height:1.1;'>{price_str}</div>"
                    f"<div style='font-size:0.76rem;color:#64748b;margin-bottom:12px;'>{range_str}</div>"
                    f"<hr style='border:none;border-top:1px solid #e5e7eb;margin:12px 0;'>"
                    f"<div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;"
                    f"letter-spacing:0.05em;'>Your cost (expected)</div>"
                    f"<div style='font-size:1.5rem;font-weight:700;color:#111;margin:2px 0;'>{cost_str}</div>"
                    f"<div style='font-size:0.76rem;color:#475569;margin-bottom:10px;'>"
                    f"Range: {cost_range_str}</div>"
                    f"<div style='font-size:0.82rem;color:{delta_col};font-weight:600;'>{delta_str}</div>"
                    f"<div style='font-size:0.82rem;color:{prob_col};margin-top:4px;'>{prob_str}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.caption(
            "Prices from Monte Carlo simulation (10,000 paths).  "
            "P10 = optimistic, P50 = median, P90 = pessimistic.  "
            "Probability uses a normal approximation over test-period residuals."
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — REGIONAL PRICE MAP
# 3-D globe / ColumnLayer — fertilizer cost exposure by US state
# ════════════════════════════════════════════════════════════════════════════
with tab_map:

    if cache is None:
        st.warning("No forecast data found. Run `python backend/run_pipeline.py` first.")
    else:
        sig_map  = cache.get("signal", {})
        cur_p    = sig_map.get("currentPrice", 0) or 1
        fc_p     = sig_map.get("forecast_t2", cur_p)
        chg_pct  = (fc_p - cur_p) / cur_p * 100
        chg_sign = "▲" if chg_pct >= 0 else "▼"

        # ── Header metrics ─────────────────────────────────────────────────
        st.subheader("🌍 Regional Fertilizer Cost Exposure")
        st.markdown(
            "Each spike shows how hard a state would be hit by a fertilizer price move. "
            "**Drag** to rotate · **scroll** to zoom · **hover** a spike for details."
        )
        with st.expander("ℹ️ How to read this map"):
            st.markdown("""
**What the spikes represent:**
Each spike sits at the geographic center of a US state. Its height and color both reflect that
state's total annual fertilizer exposure — how many metric tons of urea its farmers need to buy.

**How exposure is calculated (USDA 2023 data):**
- Corn acres × 150 lbs nitrogen/acre
- Wheat acres × 90 lbs nitrogen/acre
- Soybeans acres × 60 lbs nitrogen/acre
- Total nitrogen ÷ 46% (urea N content) ÷ 2,204.6 lbs/mt = **metric tons of urea**

**Color scale:**
| Color | Meaning |
|---|---|
| 🟢 Green | Low exposure — state doesn't grow much of these crops |
| 🟡 Yellow | Moderate exposure |
| 🔴 Red | High exposure — a price spike here hits farmers hard in absolute dollar terms |

**Why Iowa and Illinois tower above everything:**
Iowa has ~12.9 million acres of corn + 9.4 million acres of soybeans. When urea rises $30/mt,
Iowa farmers collectively face ~$57M in added fertilizer costs. That's why they light up red.

**"60-day cost impact" mode** switches height and color to show the *dollar change* driven
specifically by the current model forecast — which states gain or lose the most from this
particular price move.

**Interacting with the map:**
- Left-click + drag to pan or rotate (3-D mode)
- Scroll wheel to zoom in/out
- Hover any spike to see the state name, urea needed, current bill, and forecast impact
            """)

        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("Current Urea", f"${cur_p:.0f}/mt")
        hm2.metric("60-Day Forecast", f"${fc_p:.0f}/mt", delta=f"{chg_pct:+.1f}%")
        hm3.metric("Signal", sig_map.get("signal", "N/A"))

        # ── Build state dataset ────────────────────────────────────────────
        state_df = build_state_df(cache)

        # ── View controls ──────────────────────────────────────────────────
        vc1, vc2 = st.columns([2, 2])
        with vc1:
            view_mode = st.radio(
                "View mode",
                ["🌐 3-D Perspective", "🗺️ Flat map (top-down)"],
                horizontal=True,
                key="map_view_mode",
            )
        with vc2:
            color_metric = st.radio(
                "Color / height by",
                ["Fertilizer exposure (urea MT)", "60-day cost impact ($M)"],
                horizontal=True,
                key="map_color_metric",
            )

        # Recompute elevation & color if user chooses cost-impact mode
        if "cost impact" in color_metric:
            max_val          = state_df["impact_m"].abs().max()
            norm             = (state_df["impact_m"].abs() / max_val).clip(0, 1)
            state_df["elevation"] = (norm * 500_000).clip(1000)
            state_df["color_r"]   = (norm * 255).astype(int)
            state_df["color_g"]   = (255 - norm * 200).clip(0, 255).astype(int)
            state_df["color_b"]   = 30
            state_df["color_a"]   = 210

        layer = pdk.Layer(
            "ColumnLayer",
            data=state_df,
            get_position="[lng, lat]",
            get_elevation="elevation",
            elevation_scale=1,
            radius=55000,
            get_fill_color="[color_r, color_g, color_b, color_a]",
            pickable=True,
            auto_highlight=True,
            coverage=0.85,
        )

        if "3-D" in view_mode:
            view = pdk.ViewState(
                longitude=-96,
                latitude=36,
                zoom=3.2,
                pitch=55,
                bearing=-10,
            )
        else:
            view = pdk.ViewState(
                longitude=-96,
                latitude=39,
                zoom=3.6,
                pitch=0,
                bearing=0,
            )

        # CARTO dark-matter style — free, no Mapbox token required
        DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            map_style=DARK_STYLE,
            tooltip={
                "html": "{tooltip}",
                "style": {
                    "backgroundColor": "#0f172a",
                    "color": "#f1f5f9",
                    "fontSize": "13px",
                    "padding": "10px 14px",
                    "borderRadius": "8px",
                    "border": "1px solid #334155",
                },
            },
        )

        st.pydeck_chart(deck, width='stretch', height=600)

        # ── Color legend ───────────────────────────────────────────────────
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
            <span style="font-size:0.8rem;color:#64748b;">Low exposure</span>
            <div style="height:12px;width:200px;background:linear-gradient(to right,
                #1e7b1e, #b8a010, #ff1e1e);border-radius:4px;"></div>
            <span style="font-size:0.8rem;color:#64748b;">High exposure</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── State rankings table ───────────────────────────────────────────
        st.subheader("State Rankings — Fertilizer Exposure")
        table_df = state_df[[
            "state", "corn_acres", "wheat_acres", "soy_acres",
            "urea_mt", "current_cost_m", "forecast_cost_m", "impact_m",
        ]].copy()
        table_df["urea_mt"]          = table_df["urea_mt"].apply(lambda x: f"{x/1000:,.0f}K mt")
        table_df["current_cost_m"]   = table_df["current_cost_m"].apply(lambda x: f"${x:.1f}M")
        table_df["forecast_cost_m"]  = table_df["forecast_cost_m"].apply(lambda x: f"${x:.1f}M")
        table_df["impact_m"]         = table_df["impact_m"].apply(lambda x: f"{chg_sign}${abs(x):.1f}M")
        def _fmt_acres(x):
            return f"{x/1e6:.1f}M ac" if x >= 1e6 else f"{x/1e3:.0f}K ac"
        for col in ("corn_acres", "wheat_acres", "soy_acres"):
            table_df[col] = table_df[col].apply(_fmt_acres)
        table_df.columns = [
            "State", "Corn", "Wheat", "Soybeans",
            "Urea Needed", "Current Cost", "60-Day Forecast", f"Impact ({chg_sign})",
        ]
        st.dataframe(table_df, width='stretch', hide_index=True)
        st.caption(
            f"USDA 2023 planted acres. Urea = total N ÷ (46% N content).  "
            f"Current price: ${cur_p:.0f}/mt → 60-day forecast: ${fc_p:.0f}/mt ({chg_pct:+.1f}%)."
        )
