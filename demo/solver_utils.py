"""Shared CVXPY solve guard used by all optimization subproblems."""

import cvxpy as cp


ACCEPTED_STATUSES = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


def solve_or_skip(problem, *, context, solver=cp.CLARABEL, **solve_kwargs):
    """Solve a problem and return False when this iteration must be skipped.

    CVXPY may either report a non-optimal status or raise ``SolverError`` before
    a status is available.  Both cases are recoverable in the outer SCP loops:
    the caller keeps its last feasible iterate and moves to the next iteration.
    """
    try:
        problem.solve(solver=solver, **solve_kwargs)
    except cp.error.SolverError as exc:
        print(f"WARNING: skipping {context}; CVXPY solver error: {exc}")
        return False

    if problem.status not in ACCEPTED_STATUSES:
        print(
            f"WARNING: skipping {context}; CVXPY returned status "
            f"{problem.status!r}."
        )
        return False
    return True
