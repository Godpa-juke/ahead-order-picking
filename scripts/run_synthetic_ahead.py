#!/usr/bin/env python3
"""Run deterministic, data-free checks of AHEAD's estimator and policy wiring."""

from __future__ import annotations

import json
from types import SimpleNamespace

from rware.engine.human_assignment import available_human_assignment_strategies
from rware.engine.staging import StagingPlanner
from rware.learning.risk_model import ServiceRiskModel


def main() -> None:
    model = ServiceRiskModel(
        backend="rolling_median",
        warmup_tasks=1,
        min_group_samples=1,
        history_window=8,
    )
    observed = ((10.0, 11.0), (10.0, 14.0), (10.0, 18.0), (10.0, 12.0))
    for static_ticks, actual_ticks in observed:
        model.observe({"agent_id": 1, "static_ticks": static_ticks}, actual_ticks)
    q50, q90 = model.predict({"agent_id": 1, "static_ticks": 10.0})

    config = SimpleNamespace(
        staging_policy="learned",
        staging_early_weight=0.5,
        staging_uncertainty_weight=0.5,
        staging_eta_backend="rolling_median",
    )
    planner = StagingPlanner(SimpleNamespace(config=config))
    result = {
        "registered_early_assignment_policies": sorted(
            name
            for name in available_human_assignment_strategies()
            if name.startswith("rv_") or name == "rendezvous"
        ),
        "arrival_prediction": {"q50": q50, "q90": q90},
        "staging_candidate_cost": planner._score(
            distance=8.0, ready_q50=q50, ready_q90=q90
        ),
        "model_stats": model.stats(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
