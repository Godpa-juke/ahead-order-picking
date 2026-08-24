"""Data-free AHEAD early-assignment and staging primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from reverse_auction_assignment import solve_auction


@dataclass(frozen=True)
class Eta:
    q50: float
    q90: float

    def __post_init__(self) -> None:
        if self.q50 < 0 or self.q90 < self.q50:
            raise ValueError("ETA quantiles must satisfy 0 <= q50 <= q90")


@dataclass(frozen=True)
class Worker:
    worker_id: str


@dataclass(frozen=True)
class Request:
    request_id: str
    robot_eta: Eta
    serviceable: bool = False
    expected_service: float = 0.0
    accumulated_wait: float = 0.0


@dataclass(frozen=True)
class AheadWeights:
    robot_wait: float = 1.0
    worker_travel: float = 1.0
    worker_wait: float = 1.0
    tail_risk: float = 0.0
    service: float = 0.0
    urgency: float = 0.0
    lead_limit: float = 60.0
    travel_budget: float | None = None


@dataclass(frozen=True)
class StagingDemand:
    request_id: str
    position: tuple[int, int]
    ready_eta: Eta


WorkerEta = Callable[[tuple[str, str]], Eta]
Distance = Callable[[str, tuple[int, int]], float]


def rendezvous_cost(
    worker_eta: Eta,
    request: Request,
    weights: AheadWeights = AheadWeights(),
) -> float | None:
    """Cost one speculative worker/robot rendezvous.

    ``None`` means the en-route pairing fails a speculative-dispatch guard.
    Parked/serviceable requests bypass those guards and reduce to worker travel
    plus the configured service and urgency terms.
    """

    robot_q50 = 0.0 if request.serviceable else request.robot_eta.q50
    service_start = max(worker_eta.q50, robot_q50)
    robot_idle = service_start - robot_q50
    worker_idle = service_start - worker_eta.q50

    if not request.serviceable:
        if robot_q50 - worker_eta.q50 > weights.lead_limit:
            return None
        if weights.travel_budget is not None and worker_eta.q50 > weights.travel_budget:
            return None

    tail_lateness = max(0.0, worker_eta.q90 - max(robot_q50, worker_eta.q50))
    return (
        weights.robot_wait * robot_idle
        + weights.worker_travel * worker_eta.q50
        + weights.worker_wait * worker_idle
        + weights.tail_risk * tail_lateness
        + weights.service * request.expected_service
        - weights.urgency * request.accumulated_wait
    )


def assign_early(
    workers: Sequence[Worker],
    requests: Sequence[Request],
    eta_for_pair: WorkerEta,
    *,
    weights: AheadWeights = AheadWeights(),
) -> list[tuple[str, str]]:
    """Assign workers to parked or en-route requests one-to-one."""

    ordered_workers = sorted(workers, key=lambda item: item.worker_id)
    ordered_requests = sorted(requests, key=lambda item: item.request_id)
    if not ordered_workers or not ordered_requests:
        return []

    blocked_cost = 1e15
    matrix: list[list[float]] = []
    for worker in ordered_workers:
        row = []
        for request in ordered_requests:
            cost = rendezvous_cost(eta_for_pair((worker.worker_id, request.request_id)), request, weights)
            row.append(blocked_cost if cost is None else cost)
        matrix.append(row)

    selected = solve_auction(matrix)
    result = []
    for row_index, object_index in enumerate(selected):
        if object_index is None or matrix[row_index][object_index] >= blocked_cost:
            continue
        result.append((ordered_workers[row_index].worker_id, ordered_requests[object_index].request_id))
    return result


def staging_score(
    distance: float,
    ready_eta: Eta,
    *,
    early_weight: float = 0.5,
    uncertainty_weight: float = 0.5,
) -> float:
    """Score one idle-worker staging target; lower is better."""

    if distance < 0:
        raise ValueError("distance must be non-negative")
    early_wait = max(0.0, ready_eta.q50 - distance)
    uncertainty = max(0.0, ready_eta.q90 - ready_eta.q50)
    return distance + early_weight * early_wait + uncertainty_weight * uncertainty


def choose_staging_targets(
    workers: Sequence[Worker],
    demand: Sequence[StagingDemand],
    distance: Distance,
    *,
    early_weight: float = 0.5,
    uncertainty_weight: float = 0.5,
) -> dict[str, tuple[int, int]]:
    """Claim staging targets one-to-one from globally sorted pair scores."""

    pairs: list[tuple[float, str, str, tuple[int, int]]] = []
    for worker in workers:
        for request in demand:
            score = staging_score(
                distance(worker.worker_id, request.position),
                request.ready_eta,
                early_weight=early_weight,
                uncertainty_weight=uncertainty_weight,
            )
            pairs.append((score, worker.worker_id, request.request_id, request.position))

    chosen: dict[str, tuple[int, int]] = {}
    claimed_requests: set[str] = set()
    for _score, worker_id, request_id, position in sorted(pairs):
        if worker_id in chosen or request_id in claimed_requests:
            continue
        chosen[worker_id] = position
        claimed_requests.add(request_id)
    return chosen
