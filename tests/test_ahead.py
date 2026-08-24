import unittest

from ahead_order_picking import (
    AheadWeights,
    Eta,
    Request,
    StagingDemand,
    Worker,
    assign_early,
    choose_staging_targets,
    rendezvous_cost,
    staging_score,
)


class AheadTests(unittest.TestCase):
    def test_early_assignment_matches_arrival_times(self):
        workers = [Worker("w0"), Worker("w1")]
        requests = [Request("r0", Eta(4, 5)), Request("r1", Eta(8, 9))]
        etas = {
            ("w0", "r0"): Eta(3, 4),
            ("w0", "r1"): Eta(9, 11),
            ("w1", "r0"): Eta(8, 10),
            ("w1", "r1"): Eta(7, 8),
        }
        self.assertEqual(
            assign_early(workers, requests, etas.__getitem__),
            [("w0", "r0"), ("w1", "r1")],
        )

    def test_too_early_speculative_pair_is_blocked(self):
        request = Request("r0", Eta(20, 22), serviceable=False)
        cost = rendezvous_cost(Eta(1, 2), request, AheadWeights(lead_limit=5))
        self.assertIsNone(cost)

    def test_serviceable_request_bypasses_lead_guard(self):
        request = Request("r0", Eta(20, 22), serviceable=True)
        cost = rendezvous_cost(Eta(1, 2), request, AheadWeights(lead_limit=0))
        self.assertIsNotNone(cost)

    def test_tail_risk_penalizes_uncertain_worker_arrival(self):
        request = Request("r0", Eta(5, 5))
        weights = AheadWeights(tail_risk=1)
        stable = rendezvous_cost(Eta(5, 5), request, weights)
        uncertain = rendezvous_cost(Eta(5, 9), request, weights)
        self.assertGreater(uncertain, stable)

    def test_staging_score_prices_early_wait_and_uncertainty(self):
        base = staging_score(5, Eta(5, 5))
        early = staging_score(2, Eta(5, 5))
        uncertain = staging_score(5, Eta(5, 9))
        self.assertGreater(early, 2)
        self.assertGreater(uncertain, base)

    def test_staging_targets_are_one_to_one(self):
        workers = [Worker("w0"), Worker("w1")]
        demand = [
            StagingDemand("r0", (0, 0), Eta(2, 3)),
            StagingDemand("r1", (9, 0), Eta(2, 3)),
        ]
        positions = {"w0": (1, 0), "w1": (8, 0)}
        distance = lambda worker_id, target: abs(positions[worker_id][0] - target[0])
        chosen = choose_staging_targets(workers, demand, distance)
        self.assertEqual(chosen, {"w0": (0, 0), "w1": (9, 0)})
        self.assertEqual(len(set(chosen.values())), 2)

    def test_invalid_eta_is_rejected(self):
        with self.assertRaises(ValueError):
            Eta(3, 2)


if __name__ == "__main__":
    unittest.main()
