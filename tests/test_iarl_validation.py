"""Regression tests for the IARL validation rerun plumbing."""

from types import SimpleNamespace

import pandas as pd
import pytest

from rware.engine.arrival import ArrivalTracker, PredictionStats
from rware.engine.warehouse_engine import _split_agent_rows
from rware.learning.risk_model import ServiceRiskModel


def _tracker() -> ArrivalTracker:
    config = SimpleNamespace(
        risk_warmup_tasks=500,
        risk_retrain_interval=500,
        risk_catboost_iterations=10,
    )
    return ArrivalTracker(SimpleNamespace(config=config))


def test_staging_backend_does_not_get_replaced_by_assignment_backend():
    tracker = _tracker()
    tracker.ensure_models("static", consumer="assignment")
    tracker.ensure_models("catboost", consumer="staging")
    tracker.ensure_models("static", consumer="assignment")

    assert tracker.human_model.backend == "static"
    assert tracker.robot_model.backend == "static"
    assert tracker.staging_robot_model.backend == "catboost"


def test_consumers_share_a_robot_model_when_the_backend_matches():
    tracker = _tracker()
    tracker.ensure_models("catboost", consumer="staging")
    tracker.ensure_models("catboost", consumer="assignment")

    assert tracker.robot_model is tracker.staging_robot_model


def test_catboost_median_cold_start_has_no_uncertainty_spread():
    model = ServiceRiskModel(
        backend="catboost_median",
        warmup_tasks=100,
        min_group_samples=1,
    )
    row = {"agent_id": 1, "static_ticks": 10.0}
    for duration in (10.0, 20.0, 30.0):
        model.observe(row, duration)

    q50, q90 = model.predict(row)

    assert q50 == pytest.approx(20.0)
    assert q90 == q50


def test_catboost_median_trains_a_single_output_model():
    pytest.importorskip("catboost", reason="install the 'learning' extra to test CatBoost")
    model = ServiceRiskModel(
        backend="catboost_median",
        feature_columns=["agent_id", "static_ticks"],
        warmup_tasks=4,
        retrain_interval=4,
        catboost_iterations=3,
        min_group_samples=1,
    )
    for static, actual in ((8.0, 9.0), (10.0, 11.0), (12.0, 13.0), (14.0, 15.0)):
        model.observe({"agent_id": 1, "static_ticks": static}, actual)

    assert model.maybe_retrain()
    q50, q90 = model.predict({"agent_id": 1, "static_ticks": 10.0})

    assert model.model_ready
    assert q90 == q50


def test_prediction_stats_separates_trained_predictions():
    stats = PredictionStats()
    stats.add(q50=8.0, q90=12.0, actual=10.0, ready=False)
    stats.add(q50=11.0, q90=15.0, actual=13.0, ready=True)

    summary = stats.summary()
    assert summary["count"] == 2
    assert summary["q50_mae"] == pytest.approx(2.0)
    assert summary["q90_coverage"] == pytest.approx(1.0)
    assert summary["ready_count"] == 1
    assert summary["ready_q50_mae"] == pytest.approx(2.0)


def test_agent_table_split_uses_human_count_as_robot_offset():
    frame = pd.DataFrame({"kind": ["h1", "h2", "r1", "r2", "r3"]})

    humans, robots = _split_agent_rows(frame, human_cnt=2, robot_cnt=3)

    assert humans["kind"].tolist() == ["h1", "h2"]
    assert robots["kind"].tolist() == ["r1", "r2", "r3"]
