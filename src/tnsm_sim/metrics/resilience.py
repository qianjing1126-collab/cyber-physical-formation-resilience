from __future__ import annotations

from math import isfinite


def _window(samples: list[tuple[float, float]], start: float, end: float) -> list[tuple[float, float]]:
    selected = [(t, value) for t, value in samples if start - 1e-12 <= t <= end + 1e-12]
    if len(selected) < 2 or abs(selected[0][0] - start) > 1e-9 or abs(selected[-1][0] - end) > 1e-9:
        raise ValueError("samples must include both fixed-window endpoints")
    return selected


def fixed_window_area(samples: list[tuple[float, float]], start: float, duration: float) -> float:
    if duration <= 0:
        raise ValueError("duration must be positive")
    selected = _window(samples, start, start + duration)
    area = sum((t1 - t0) * (y0 + y1) / 2 for (t0, y0), (t1, y1) in zip(selected, selected[1:]))
    return area / duration


def recovery_time(
    q_samples: list[tuple[float, float]],
    t_d: float,
    T_e: float,
    T_h: float,
    q_target: float,
) -> tuple[float | None, bool]:
    selected = _window(q_samples, t_d, t_d + T_e)
    for index, (start, value) in enumerate(selected):
        if value + 1e-12 < q_target or start > t_d + T_e - T_h + 1e-12:
            continue
        end_required = start + T_h
        covered = False
        valid = True
        for t, q in selected[index:]:
            if t > end_required + 1e-12:
                break
            if q + 1e-12 < q_target:
                valid = False
                break
            if t >= end_required - 1e-12:
                covered = True
                break
        if valid and covered:
            return start - t_d, False
    return None, True


def minimum_q(samples: list[tuple[float, float]], start: float, duration: float) -> float:
    return min(value for _, value in _window(samples, start, start + duration))


def mission_area(samples: list[tuple[float, float]]) -> float:
    if len(samples) < 2:
        return float("nan")
    duration = samples[-1][0] - samples[0][0]
    if duration <= 0:
        return float("nan")
    return sum((t1 - t0) * (y0 + y1) / 2 for (t0, y0), (t1, y1) in zip(samples, samples[1:])) / duration

