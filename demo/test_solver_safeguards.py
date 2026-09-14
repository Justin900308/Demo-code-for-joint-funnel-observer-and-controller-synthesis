import unittest

import cvxpy as cp

from solver_utils import solve_or_skip


class FakeProblem:
    def __init__(self, status=None, exception=None):
        self.status = status
        self.exception = exception

    def solve(self, **_kwargs):
        if self.exception is not None:
            raise self.exception


class SolverSafeguardTests(unittest.TestCase):
    def test_infeasible_status_is_skipped(self):
        problem = FakeProblem(status=cp.INFEASIBLE)
        self.assertFalse(solve_or_skip(problem, context="test iteration"))

    def test_solver_error_is_skipped(self):
        problem = FakeProblem(exception=cp.error.SolverError("test failure"))
        self.assertFalse(solve_or_skip(problem, context="test iteration"))

    def test_optimal_status_is_accepted(self):
        problem = FakeProblem(status=cp.OPTIMAL)
        self.assertTrue(solve_or_skip(problem, context="test iteration"))


if __name__ == "__main__":
    unittest.main()
