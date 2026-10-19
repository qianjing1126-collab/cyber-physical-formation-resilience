from __future__ import annotations

from time import perf_counter

import cvxpy as cp


ACCEPTED_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


def solve_problem(problem: cp.Problem, solver: str, time_limit_s: float) -> dict:
    solver = solver.upper()
    options: dict[str, object]
    if solver == "CLARABEL":
        options = {
            "max_iter": 100_000,
            "time_limit": time_limit_s,
            "tol_gap_abs": 1e-8,
            "tol_gap_rel": 1e-8,
            "tol_feas": 1e-8,
        }
    elif solver == "SCS":
        options = {
            "max_iters": 200_000,
            "time_limit_secs": time_limit_s,
            "eps_abs": 1e-6,
            "eps_rel": 1e-6,
            "eps_infeas": 1e-7,
        }
    else:
        raise ValueError(f"unsupported SDP solver: {solver}")
    started = perf_counter()
    try:
        value = problem.solve(solver=solver, verbose=False, **options)
        error = None
    except Exception as exc:  # preserved in the immutable solver record
        value = None
        error = f"{type(exc).__name__}: {exc}"
    wall_time = perf_counter() - started
    stats = problem.solver_stats
    return {
        "solver": solver,
        "status": problem.status,
        "accepted_status": problem.status in ACCEPTED_STATUSES,
        "objective": None if value is None else float(value),
        "wall_time_s": wall_time,
        "solve_time_s": None if stats is None else stats.solve_time,
        "setup_time_s": None if stats is None else stats.setup_time,
        "num_iters": None if stats is None else stats.num_iters,
        "error": error,
    }
