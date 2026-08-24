# AHEAD: Anticipatory Human Early Assignment and Deployment

A data-free reference implementation of two timing interventions for collaborative human–robot order picking.

- **AHEAD-A — early assignment:** consider robot requests while the robot is still traveling and match predicted worker and robot arrival times.
- **AHEAD-D — deployment:** move otherwise-idle workers toward open demand before a request becomes serviceable.

This repository contains only reusable algorithm code, explanatory text, and toy invariant tests. It does not publish a paper, experiment output, warehouse workload, or measured result.

## Relationship to the baseline

AHEAD-A uses the one-to-one solver from [`reverse-auction-order-picking`](https://github.com/Godpa-juke/reverse-auction-order-picking), pinned to a specific commit in `pyproject.toml`.

## What is included

- quantile ETA input type (`q50`, `q90`);
- rendezvous cost with robot wait, worker travel, worker wait, and tail-risk terms;
- guards for speculative en-route assignments;
- one-to-one early assignment;
- one-to-one idle-worker staging with deterministic tie-breaking;
- toy tests for timing and assignment invariants.

## What is not included

- PDF, manuscript, LaTeX, tables, or generated figures;
- measured performance values or experiment summaries;
- raw or aggregate data;
- warehouse layouts, orders, models, or model checkpoints;
- private simulator code or Git history.

## Install

```bash
python -m pip install -e .
```

## AHEAD-A example

```python
from ahead_order_picking import Eta, Request, Worker, assign_early

workers = [Worker("w0"), Worker("w1")]
requests = [Request("r0", robot_eta=Eta(4, 5)), Request("r1", robot_eta=Eta(8, 9))]

worker_eta = {
    ("w0", "r0"): Eta(3, 4),
    ("w0", "r1"): Eta(9, 11),
    ("w1", "r0"): Eta(8, 10),
    ("w1", "r1"): Eta(7, 8),
}

print(assign_early(workers, requests, worker_eta.__getitem__))
```

## AHEAD-D staging score

For an idle worker at distance `d` from an open request with predicted readiness interval `[q50, q90]`, the reference score is:

```text
d
+ early_weight × max(0, q50 - d)
+ uncertainty_weight × max(0, q90 - q50)
```

Lower is better. Requests and workers are claimed one-to-one in deterministic global score order.

## Test

```bash
python -m unittest discover -s tests -v
```
