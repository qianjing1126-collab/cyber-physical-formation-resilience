from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _read_json_yaml(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} must use the JSON subset of YAML during G1: {exc}"
        ) from exc


def _load_recursive(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        raise ValueError(f"cyclic config inheritance: {path}")
    data = _read_json_yaml(path)
    parent = data.get("extends")
    if not parent:
        return data
    base = _load_recursive((path.parent / parent).resolve(), stack + (path,))
    return _deep_merge(base, data)


def _require(cfg: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise ValueError(f"missing required configuration field: {dotted}")
        cur = cur[part]
    return cur


def _positive(cfg: Mapping[str, Any], dotted: str, *, allow_zero: bool = False) -> float:
    value = float(_require(cfg, dotted))
    if (allow_zero and value < 0) or (not allow_zero and value <= 0):
        raise ValueError(f"{dotted} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def validate_config(cfg: Mapping[str, Any]) -> None:
    groups = ("meta", "time", "vehicle", "path", "formation", "network", "control", "failure", "recovery", "output")
    for group in groups:
        _require(cfg, group)
    for field in (
        "time.dt_dyn", "time.T_ctrl", "time.T_msg", "vehicle.N",
        "vehicle.body_length", "vehicle.body_width", "vehicle.plow_width",
        "vehicle.v_ref", "vehicle.a_x_max", "vehicle.a_y_max",
        "formation.slot_spacing", "network.packet_size", "network.rate_hz",
        "recovery.T_e", "recovery.T_h", "output.state_period",
        "output.metric_period",
    ):
        _positive(cfg, field)
    if int(_require(cfg, "vehicle.N")) < 2:
        raise ValueError("vehicle.N must be at least 2")
    path_id = _require(cfg, "path.path_id")
    if path_id not in {"P1", "P2", "P3"}:
        raise ValueError("path.path_id must be P1, P2 or P3")
    if path_id == "P2":
        _positive(cfg, "path.radius")
    if path_id == "P3":
        for field in ("path.transition_start", "path.transition_length"):
            _positive(cfg, field, allow_zero=(field == "path.transition_start"))
        _require(cfg, "path.lateral_shift")
        if float(_require(cfg, "path.transition_start")) + float(_require(cfg, "path.transition_length")) >= float(_require(cfg, "path.length")):
            raise ValueError("P3 transition must leave a final straight segment")
    layout_id = _require(cfg, "formation.layout_id")
    if layout_id not in {"F1", "F2", "F3"}:
        raise ValueError("formation.layout_id must be F1, F2 or F3")
    if layout_id == "F2":
        offsets = tuple(float(value) for value in _require(cfg, "formation.longitudinal_offsets"))
        if len(offsets) != int(_require(cfg, "vehicle.N")):
            raise ValueError("F2 longitudinal_offsets must match vehicle.N")
        if max(offsets) - min(offsets) > float(_require(cfg, "formation.epsilon_parallel")) + 1e-12:
            raise ValueError("F2 longitudinal offsets exceed epsilon_parallel")
    if layout_id == "F3":
        lower = float(_require(cfg, "formation.heterogeneity_min"))
        upper = float(_require(cfg, "formation.heterogeneity_max"))
        if not 0 < lower <= upper:
            raise ValueError("invalid F3 heterogeneity range")
    dt = float(_require(cfg, "time.dt_dyn"))
    for field in ("time.T_ctrl", "time.T_msg", "output.state_period", "output.metric_period"):
        ratio = float(_require(cfg, field)) / dt
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"{field} must be an integer multiple of time.dt_dyn")
    start = float(_require(cfg, "time.t_start"))
    end = float(_require(cfg, "time.t_end"))
    if end <= start:
        raise ValueError("time.t_end must be greater than time.t_start")
    if float(_require(cfg, "recovery.T_h")) > float(_require(cfg, "recovery.T_e")):
        raise ValueError("recovery.T_h cannot exceed recovery.T_e")
    zeta = float(_require(cfg, "recovery.zeta_r"))
    if not 0 < zeta <= 1:
        raise ValueError("recovery.zeta_r must lie in (0, 1]")
    spacing = float(_require(cfg, "formation.slot_spacing"))
    body_width = float(_require(cfg, "vehicle.body_width"))
    clearance = float(_require(cfg, "recovery.safety_clearance"))
    if spacing + 1e-12 < body_width + clearance:
        raise ValueError("nominal slot spacing violates the configured safety clearance")
    failure_type = _require(cfg, "failure.type")
    targets = list(_require(cfg, "failure.targets"))
    if failure_type == "none" and targets:
        raise ValueError("failure.targets must be empty when failure.type is none")
    if failure_type != "none":
        td = _require(cfg, "failure.t_d")
        tc = _require(cfg, "failure.t_c")
        if td is None or tc is None or not (start <= float(td) <= float(tc) <= end):
            raise ValueError("a physical failure requires t_start <= t_d <= t_c <= t_end")
        if any(int(x) < 1 or int(x) > int(_require(cfg, "vehicle.N")) for x in targets):
            raise ValueError("failure target is outside 1..N")
    if bool(_require(cfg, "recovery.enabled")) and failure_type == "none":
        raise ValueError("recovery cannot be enabled without a physical failure")
    recovery_method = str(cfg.get("recovery", {}).get("method", "S3")).upper()
    if recovery_method not in {"B0", "B9", "B11", "S3"}:
        raise ValueError("recovery.method must be B0, B9, B11 or S3")
    if bool(_require(cfg, "recovery.enabled")) and recovery_method == "B0":
        raise ValueError("B0 must keep recovery.enabled=false")
    backend = _require(cfg, "network.backend")
    if backend not in {"ideal", "packet_event"}:
        raise ValueError("G1 network.backend must be ideal or packet_event")
    if backend == "packet_event":
        for field in ("network.access_delay", "network.propagation_delay", "network.jitter", "network.loss_probability"):
            _positive(cfg, field, allow_zero=True)
        loss = float(_require(cfg, "network.loss_probability"))
        if loss > 1:
            raise ValueError("network.loss_probability cannot exceed 1")
        delay_cdf = cfg.get("network", {}).get("delay_cdf")
        if delay_cdf is not None:
            probabilities = delay_cdf.get("probabilities", ())
            quantiles = delay_cdf.get("quantiles_s", ())
            if len(probabilities) != len(quantiles) or len(probabilities) < 2:
                raise ValueError("network.delay_cdf must contain aligned probabilities and quantiles_s")
        two_state = cfg.get("network", {}).get("two_state")
        if two_state is not None:
            if delay_cdf is not None:
                raise ValueError("network.delay_cdf and network.two_state are mutually exclusive")
            if two_state.get("model") != "two_state_markov_shared_broadcast":
                raise ValueError("unsupported network.two_state model")
            transition = two_state.get("transition_matrix", ())
            if len(transition) != 2 or any(len(row) != 2 for row in transition):
                raise ValueError("network.two_state.transition_matrix must be 2x2")
            if set(two_state.get("states", {})) != {"clear", "contended"}:
                raise ValueError("network.two_state requires clear and contended states")
    logical_edges = cfg.get("network", {}).get("logical_edges")
    if logical_edges is not None:
        n = int(_require(cfg, "vehicle.N"))
        normalized: set[tuple[int, int]] = set()
        for edge in logical_edges:
            if len(edge) != 2:
                raise ValueError("network.logical_edges entries must contain two node ids")
            i, j = int(edge[0]), int(edge[1])
            if i == j or not (1 <= i <= n and 1 <= j <= n):
                raise ValueError("network.logical_edges must define simple edges inside 1..N")
            normalized.add((min(i, j), max(i, j)))
        if len(normalized) != len(logical_edges):
            raise ValueError("network.logical_edges must be duplicate-free")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(val) for key, val in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(x) for x in value)
    return value


@dataclass(frozen=True)
class FrozenConfig:
    data: Mapping[str, Any]
    canonical_json: str
    config_hash: str
    source: Path

    def section(self, name: str) -> Mapping[str, Any]:
        return self.data[name]


def load_config(path: str | Path) -> FrozenConfig:
    source = Path(path).resolve()
    data = _load_recursive(source)
    network = data.get("network", {})
    profile_ref = network.get("two_state_profile_file")
    if profile_ref is not None:
        regime = network.get("two_state_regime")
        if not regime:
            raise ValueError("network.two_state_regime is required with two_state_profile_file")
        profile_path = (source.parent / str(profile_ref)).resolve()
        profiles = _read_json_yaml(profile_path)
        if regime not in profiles:
            raise ValueError(f"two-state regime {regime!r} is absent from {profile_path}")
        network["two_state"] = profiles[regime]
    validate_config(data)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return FrozenConfig(
        data=_freeze(data),
        canonical_json=canonical,
        config_hash=sha256(canonical.encode("utf-8")).hexdigest(),
        source=source,
    )
