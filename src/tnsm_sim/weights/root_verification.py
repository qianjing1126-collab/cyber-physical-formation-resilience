from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from numpy.polynomial import Polynomial

from .delay_margin import modal_delay_margin


def _pade_coefficients(order: int) -> np.ndarray:
    if order < 1:
        raise ValueError("order must be positive")
    return np.asarray([
        factorial(2 * order - k) * factorial(order)
        / (factorial(2 * order) * factorial(k) * factorial(order - k))
        for k in range(order + 1)
    ], dtype=float)


def pade_characteristic_roots(
    delay: float,
    lam: float,
    k_p: float,
    k_v: float,
    order: int = 8,
) -> np.ndarray:
    """Roots after an [order/order] Padé approximation of exp(-s*delay)."""
    if delay <= 0:
        roots = np.roots([1.0, lam * k_v, lam * k_p])
        return np.asarray(roots, dtype=complex)
    coefficients = _pade_coefficients(order)
    powers = np.asarray([delay**k for k in range(order + 1)])
    denominator = Polynomial(coefficients * powers)
    numerator = Polynomial(coefficients * powers * np.asarray([(-1.0) ** k for k in range(order + 1)]))
    s = Polynomial([0.0, 1.0])
    characteristic = s * s * denominator + lam * (k_v * s + k_p) * numerator
    return characteristic.roots()


def spectral_abscissa(delay: float, lam: float, k_p: float, k_v: float, order: int = 8) -> float:
    roots = pade_characteristic_roots(delay, lam, k_p, k_v, order)
    return float(np.max(roots.real))


def pade_delay_margin(
    lam: float,
    k_p: float,
    k_v: float,
    order: int = 8,
    tolerance: float = 1e-8,
) -> float:
    analytic = modal_delay_margin(lam, k_p, k_v)
    low, high = 0.5 * analytic, 1.5 * analytic
    while spectral_abscissa(low, lam, k_p, k_v, order) >= 0:
        low *= 0.5
    while spectral_abscissa(high, lam, k_p, k_v, order) <= 0:
        high *= 1.5
    while high - low > tolerance:
        middle = (low + high) / 2
        if spectral_abscissa(middle, lam, k_p, k_v, order) < 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


@dataclass(frozen=True)
class DDESimulationResult:
    delay: float
    growth_rate: float
    final_amplitude: float
    peak_amplitude: float


def simulate_delay_mode(
    delay: float,
    lam: float,
    k_p: float,
    k_v: float,
    dt: float = 0.002,
    duration: float = 25.0,
) -> DDESimulationResult:
    """Independent method-of-steps RK4 simulation with constant prehistory."""
    if delay <= dt:
        raise ValueError("delay must be greater than dt for the delayed RK4 evaluator")
    steps = int(round(duration / dt)) + 1
    time = np.arange(steps, dtype=float) * dt
    position = np.empty(steps, dtype=float)
    velocity = np.empty(steps, dtype=float)
    position[0], velocity[0] = 1.0, 0.0

    def history(query: float, known_index: int) -> tuple[float, float]:
        if query <= 0:
            return 1.0, 0.0
        scaled = query / dt
        left = min(int(scaled), known_index - 1)
        fraction = scaled - left
        return (
            position[left] * (1 - fraction) + position[left + 1] * fraction,
            velocity[left] * (1 - fraction) + velocity[left + 1] * fraction,
        )

    def acceleration(query_time: float, known_index: int) -> float:
        delayed_position, delayed_velocity = history(query_time - delay, known_index)
        return -lam * (k_v * delayed_velocity + k_p * delayed_position)

    for index in range(steps - 1):
        now = time[index]
        k1_position = velocity[index]
        k1_velocity = acceleration(now, index)
        k2_position = velocity[index] + 0.5 * dt * k1_velocity
        k2_velocity = acceleration(now + 0.5 * dt, index)
        k3_position = velocity[index] + 0.5 * dt * k2_velocity
        k3_velocity = acceleration(now + 0.5 * dt, index)
        k4_position = velocity[index] + dt * k3_velocity
        k4_velocity = acceleration(now + dt, index)
        position[index + 1] = position[index] + dt * (
            k1_position + 2 * k2_position + 2 * k3_position + k4_position
        ) / 6
        velocity[index + 1] = velocity[index] + dt * (
            k1_velocity + 2 * k2_velocity + 2 * k3_velocity + k4_velocity
        ) / 6

    omega_scale = np.sqrt(max(lam * k_p, 1e-12))
    amplitude = np.sqrt(position**2 + (velocity / omega_scale) ** 2)
    tail = time >= duration * 0.5
    growth_rate = float(np.polyfit(time[tail], np.log(np.maximum(amplitude[tail], 1e-14)), 1)[0])
    return DDESimulationResult(delay, growth_rate, float(amplitude[-1]), float(np.max(amplitude)))


def empirical_delay_margin(
    lam: float,
    k_p: float,
    k_v: float,
    dt: float = 0.002,
    duration: float = 25.0,
    tolerance: float = 2e-4,
) -> float:
    analytic = modal_delay_margin(lam, k_p, k_v)
    low, high = 0.8 * analytic, 1.2 * analytic
    while high - low > tolerance:
        middle = (low + high) / 2
        result = simulate_delay_mode(middle, lam, k_p, k_v, dt, duration)
        if result.growth_rate < 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2
