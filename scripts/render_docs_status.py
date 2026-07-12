"""
Render a small hand-written status page from dbt artifacts, because the
standard dbt-docs static site (index.html/manifest.json/catalog.json) does
not surface `dbt test` or `dbt source freshness` results anywhere in its UI
— both are dbt Cloud-only features. Confirmed against dbt-labs/dbt-docs#576.

Reads run_results.json (from `dbt test`) and sources.json (from
`dbt source freshness`), cross-references manifest.json for readable names,
and writes two self-contained HTML files:
  - index.html   landing page with a one-line pass/fail summary banner,
                  linking to the real dbt docs and to results.html
  - results.html full table of test and freshness results

Missing artifacts (e.g. a step was skipped) render as an empty section
rather than erroring, since the run_results/sources.json files are
expected to exist even when steps report failures upstream via
continue-on-error.

Usage (from project root):
    python scripts/render_docs_status.py \
        --run-results foregast_dbt/target/run_results.json \
        --sources foregast_dbt/target/sources.json \
        --manifest foregast_dbt/target/manifest.json \
        --out-dir _site
"""

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

PASS_STATUSES = {"pass", "success"}
WARN_STATUSES = {"warn"}
FAIL_STATUSES = {"fail", "error", "runtime error"}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _node_name(manifest: dict, unique_id: str) -> str:
    for key in ("nodes", "sources"):
        node = manifest.get(key, {}).get(unique_id)
        if node:
            return node.get("name", unique_id)
    return unique_id


def _status_class(status: str) -> str:
    status = (status or "").lower()
    if status in PASS_STATUSES:
        return "ok"
    if status in WARN_STATUSES:
        return "warn"
    if status in FAIL_STATUSES:
        return "bad"
    return "unknown"


def _test_rows(run_results: dict, manifest: dict) -> list[dict]:
    rows = []
    for r in run_results.get("results", []):
        uid = r.get("unique_id", "")
        if not uid.startswith("test."):
            continue
        rows.append(
            {
                "name": _node_name(manifest, uid),
                "status": r.get("status", "unknown"),
                "message": r.get("message") or "",
                "execution_time": r.get("execution_time", 0.0),
            }
        )
    rows.sort(key=lambda r: (_status_class(r["status"]) != "bad", r["name"]))
    return rows


def _freshness_rows(sources: dict, manifest: dict) -> list[dict]:
    rows = []
    for r in sources.get("results", []):
        uid = r.get("unique_id", "")
        criteria = r.get("criteria", {}) or {}
        rows.append(
            {
                "name": _node_name(manifest, uid),
                "status": r.get("status", "unknown"),
                "max_loaded_at": r.get("max_loaded_at", ""),
                "age_hours": round((r.get("max_loaded_at_time_ago_in_s") or 0) / 3600, 1),
                "warn_after": criteria.get("warn_after"),
                "error_after": criteria.get("error_after"),
            }
        )
    rows.sort(key=lambda r: (_status_class(r["status"]) != "bad", r["name"]))
    return rows


def _summarize(rows: list[dict]) -> dict:
    counts = {"ok": 0, "warn": 0, "bad": 0, "unknown": 0}
    for r in rows:
        counts[_status_class(r["status"])] += 1
    return counts


def _badge(status: str) -> str:
    cls = _status_class(status)
    return f'<span class="badge {cls}">{html.escape(status)}</span>'


PAGE_CSS = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 2rem auto; max-width: 900px; color: #1a1a1a; }
h1 { font-size: 1.4rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #e0e0e0; font-size: 0.9rem; }
th { background: #f5f5f5; }
.badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 3px; font-size: 0.8rem; font-weight: 600; }
.badge.ok { background: #d4edda; color: #155724; }
.badge.warn { background: #fff3cd; color: #856404; }
.badge.bad { background: #f8d7da; color: #721c24; }
.badge.unknown { background: #e2e3e5; color: #383d41; }
.summary { padding: 0.75rem 1rem; border-radius: 4px; background: #f5f5f5; margin: 1rem 0; }
.summary.bad { background: #f8d7da; }
a { color: #0b5ed7; }
.links a { display: inline-block; margin-right: 1.5rem; font-weight: 600; }
.muted { color: #666; font-size: 0.85rem; }
"""


def render_index(test_counts: dict, freshness_counts: dict, generated_at: str) -> str:
    overall_bad = test_counts["bad"] > 0 or freshness_counts["bad"] > 0
    overall_class = "bad" if overall_bad else ("warn" if freshness_counts["warn"] else "ok")
    summary = (
        f"Tests: {test_counts['ok']} passed, {test_counts['bad']} failed"
        f" &middot; Freshness: {freshness_counts['ok']} pass, "
        f"{freshness_counts['warn']} warn, {freshness_counts['bad']} error"
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>foreGASt — dbt docs</title><style>{PAGE_CSS}</style></head>
<body>
<h1>foreGASt dbt docs</h1>
<div class="summary {overall_class}">{summary}<br>
<span class="muted">Generated {html.escape(generated_at)}</span></div>
<div class="links">
<a href="docs/index.html">Browse dbt docs &rarr;</a>
<a href="results.html">Full test &amp; freshness details &rarr;</a>
</div>
</body></html>"""


def render_results(test_rows: list[dict], freshness_rows: list[dict], generated_at: str) -> str:
    test_html = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{_badge(r['status'])}</td>"
        f"<td>{html.escape(r['message'])}</td><td>{r['execution_time']:.2f}s</td></tr>"
        for r in test_rows
    ) or '<tr><td colspan="4" class="muted">No test results (run_results.json missing or empty).</td></tr>'

    fresh_html = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{_badge(r['status'])}</td>"
        f"<td>{html.escape(str(r['max_loaded_at']))}</td><td>{r['age_hours']}h</td>"
        f"<td>warn {html.escape(str(r['warn_after']))} / error {html.escape(str(r['error_after']))}</td></tr>"
        for r in freshness_rows
    ) or '<tr><td colspan="5" class="muted">No freshness results (sources.json missing, empty, or no sources have loaded_at_field configured).</td></tr>'

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>foreGASt — test &amp; freshness results</title><style>{PAGE_CSS}</style></head>
<body>
<p><a href="index.html">&larr; back</a></p>
<h1>Test &amp; freshness results</h1>
<p class="muted">Generated {html.escape(generated_at)}</p>

<h2>dbt test</h2>
<table><thead><tr><th>Test</th><th>Status</th><th>Message</th><th>Time</th></tr></thead>
<tbody>{test_html}</tbody></table>

<h2>dbt source freshness</h2>
<table><thead><tr><th>Source</th><th>Status</th><th>Max loaded at</th><th>Age</th><th>Thresholds</th></tr></thead>
<tbody>{fresh_html}</tbody></table>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-results", required=True, type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    run_results = _load_json(args.run_results)
    sources = _load_json(args.sources)

    test_rows = _test_rows(run_results, manifest)
    freshness_rows = _freshness_rows(sources, manifest)
    test_counts = _summarize(test_rows)
    freshness_counts = _summarize(freshness_rows)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "index.html").write_text(
        render_index(test_counts, freshness_counts, generated_at), encoding="utf-8"
    )
    (args.out_dir / "results.html").write_text(
        render_results(test_rows, freshness_rows, generated_at), encoding="utf-8"
    )

    print(f"tests: {test_counts}  freshness: {freshness_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
