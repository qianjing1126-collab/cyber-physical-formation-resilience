from __future__ import annotations

from math import atan, sqrt


def critical_frequency(lam: float, k_p: float, k_v: float) -> float:
    if lam <= 0 or k_p <= 0 or k_v <= 0:
        raise ValueError("lambda, k_p, and k_v must be positive")
    omega_sq = (lam**2 * k_v**2 + sqrt(lam**4 * k_v**4 + 4 * lam**2 * k_p**2)) / 2
    return sqrt(omega_sq)


def modal_delay_margin(lam: float, k_p: float, k_v: float) -> float:
    omega = critical_frequency(lam, k_p, k_v)
    return atan(k_v * omega / k_p) / omega


def lambda_cap_for_delay_margin(target_delay: float, k_p: float, k_v: float) -> float:
    """Invert the monotone modal delay-margin relation without a fitted constant."""
    if target_delay <= 0 or k_p <= 0 or k_v <= 0:
        raise ValueError("target_delay, k_p, and k_v must be positive")
    lower, upper = 1e-10, 1.0
    while modal_delay_margin(upper, k_p, k_v) > target_delay:
        upper *= 2.0
        if upper > 1e12:
            raise RuntimeError("could not bracket the spectral cap")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if modal_delay_margin(midpoint, k_p, k_v) > target_delay:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)
