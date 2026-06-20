"""Pydantic v2 response models mirroring data/processed/cache.json.

Wire-format normalisations vs. the on-disk cache:
  * signal.{currentPrice,bestMonth,bestPrice} → snake_case on the wire.
  * signal.as_of_date is dropped — top-level as_of_date is canonical.
  * forecast gains months: list[str] (ISO YYYY-MM) alongside display labels.
  * Envelope carries schema_version: 1.
"""

from __future__ import annotations

from typing import Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class LabeledSeries(_Base):
    labels: list[str]
    values: list[float]


class ForecastBand(_Base):
    labels: list[str] = Field(description="Display labels, e.g. 'Jun 2026'.")
    months: list[str] = Field(description="ISO YYYY-MM strings derived from the anchor (anchor+1..+N).")
    mean:   list[float]
    low:    list[float]
    high:   list[float]


class SignalPayload(_Base):
    signal:           str   = Field(description="BUY_NOW | CONSIDER_BUYING | WAIT | NEUTRAL")
    urgency:          str   = Field(description="HIGH | MODERATE | LOW")
    recommendation:   str
    rationale:        str
    key_driver:       str
    forecast_summary: str
    confidence:       float = Field(
        description="Probability driving the call. For WAIT this is (1 - prob_rising); "
                    "for BUY/CONSIDER/NEUTRAL it equals prob_rising."
    )
    current_price:    float = Field(validation_alias="currentPrice")
    best_month:       str   = Field(validation_alias="bestMonth")
    best_price:       float = Field(validation_alias="bestPrice")
    forecast_t1:      float
    forecast_t2:      float
    forecast_t3:      float
    p10_t2:           float
    p90_t2:           float
    prob_rising:      float
    ng_current:       float
    ng_change_30d:    float


class MonteCarloStats(_Base):
    n_simulations: int
    p10_t1: float
    p25_t1: float
    p50_t1: float
    p75_t1: float
    p90_t1: float
    p10_t2: float
    p25_t2: float
    p50_t2: float
    p75_t2: float
    p90_t2: float
    p10_t3: float
    p25_t3: Optional[float] = Field(default=None, description="Not produced by current pipeline.")
    p50_t3: float
    p75_t3: Optional[float] = Field(default=None, description="Not produced by current pipeline.")
    p90_t3: float
    prob_rising_t2:         float
    prob_10pct_increase_t2: float
    prob_20pct_increase_t2: float


class ModelMetadata(_Base):
    training_window_start: str
    training_window_end:   str
    test_rmse_t1: float
    test_mae_t1:  float
    test_dir_acc_t1: float
    test_clf_acc_t1: float
    residual_mean_t1: float
    residual_std_t1:  float
    test_rmse_t2: float
    test_mae_t2:  float
    test_dir_acc_t2: float
    test_clf_acc_t2: float
    residual_mean_t2: float
    residual_std_t2:  float
    test_rmse_t3: float
    test_mae_t3:  float
    test_dir_acc_t3: float
    test_clf_acc_t3: float
    residual_mean_t3: float
    residual_std_t3:  float


class ForecastBundle(_Base):
    schema_version: int = 1
    generated_at:   AwareDatetime
    as_of_date:     str = Field(description="Anchor month (YYYY-MM). First forecast month is anchor+1.")
    urea_history:        LabeledSeries
    natgas_history:      LabeledSeries
    forecast:            ForecastBand
    signal:              SignalPayload
    monte_carlo:         MonteCarloStats
    sim_t2_distribution: list[float]
    model_metadata:      ModelMetadata


class ForecastSummary(_Base):
    schema_version: int = 1
    generated_at:   AwareDatetime
    as_of_date:     str
    forecast:       ForecastBand
    signal:         SignalPayload


class MonteCarloBundle(_Base):
    schema_version:      int = 1
    monte_carlo:         MonteCarloStats
    sim_t2_distribution: list[float]


class HealthResponse(_Base):
    status:            str
    schema_version:    int = 1
    cache_age_seconds: int
    as_of_date:        str = Field(description="Anchor month in the cache (YYYY-MM).")
    data_through:      str = Field(
        description="Last month present in the cache's history series (YYYY-MM). "
                    "Equals as_of_date today because run_pipeline intersects NG and urea; "
                    "may exceed it if the producer later exposes the trailing NG-only month."
    )
