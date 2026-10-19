"""Reproduce the G8 pure recovery-plan-construction timing table."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from tnsm_sim.experiments.artifacts import ArtifactStore
from tnsm_sim.experiments.v21_g8 import benchmark_plan_construction


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = json.loads((ROOT / "configs" / "v2_1_g8.json").read_text(encoding="utf-8"))
    config = deepcopy(config)
    config["run_name"] = "V23_G8_plan_construction_grouped"
    raw, summary = benchmark_plan_construction(config)
    store = ArtifactStore(ROOT / "runs" / "v23_g8_plan_construction", config["run_name"])
    store.write_json("protocol.json", config)
    store.write_csv("plan_construction_raw.csv", raw, list(raw[0]))
    store.write_csv("plan_construction_summary.csv", summary, list(summary[0]))
    print(store.finalize())


if __name__ == "__main__":
    main()
