from __future__ import annotations

import argparse
import json
from pathlib import Path

from tnsm_sim.config import load_config
from tnsm_sim.experiments import analyze_v21_g2, build_g4_matrix, build_v21_g2_matrix, run_g3_network_calibration, run_g3_solver_gate, run_g3b2_calibration, run_g3b3_calibration, run_g4_batch, run_p01, run_p02, run_v21_g1, run_v21_g2_batch, run_v21_g3, run_v21_g5, run_v21_g6, run_v21_g7, run_v21_g8, run_v21_g9
from tnsm_sim.io import verify_manifest
from tnsm_sim.plotting import generate_g1_overview
from tnsm_sim.runner import run_simulation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tnsm-sim")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("config", type=Path)
    run = sub.add_parser("run")
    run.add_argument("config", type=Path)
    run.add_argument("--plot", action="store_true", help="generate figures from the completed raw run")
    run.add_argument("--show", action="store_true", help="open the generated overview window (implies --plot)")
    audit = sub.add_parser("audit")
    audit.add_argument("run_dir", type=Path)
    plot = sub.add_parser("plot")
    plot.add_argument("run_dir", type=Path)
    plot.add_argument("--show", action="store_true")
    p01 = sub.add_parser("g2-p01", help="run the Stage 1 structural-certification pilot")
    p01.add_argument("config", type=Path)
    p01.add_argument("--output-root", type=Path, default=Path("runs") / "g2")
    p02 = sub.add_parser("g2-p02", help="run the Stage 2 delay-margin pilot")
    p02.add_argument("config", type=Path)
    p02.add_argument("--output-root", type=Path, default=Path("runs") / "g2")
    g3 = sub.add_parser("g3-opt", help="run the formal Stage 1/2 solver-certification gate")
    g3.add_argument("policy", type=Path)
    g3.add_argument("--output-root", type=Path, default=Path("runs") / "g3")
    g3_network = sub.add_parser("g3-net-calibrate", help="fit and holdout-test the packet-event adapter against ns-3")
    g3_network.add_argument("config", type=Path)
    g3_network.add_argument("trace_root", type=Path)
    g3_network.add_argument("--output-root", type=Path, default=Path("runs") / "g3")
    g3b2 = sub.add_parser("g3-net-calibrate-b2", help="fit and holdout-test the two-state correlated-contention adapter")
    g3b2.add_argument("config", type=Path)
    g3b2.add_argument("trace_root", type=Path)
    g3b2.add_argument("--output-root", type=Path, default=Path("runs") / "g3")
    g3b3 = sub.add_parser("g3-net-calibrate-b3", help="fit and holdout-test against the stationary jittered ns-3 reference")
    g3b3.add_argument("config", type=Path)
    g3b3.add_argument("trace_root", type=Path)
    g3b3.add_argument("--output-root", type=Path, default=Path("runs") / "g3")
    g4_build = sub.add_parser("g4-build", help="freeze the 900-run G4 reportable matrix")
    g4_run = sub.add_parser("g4-run", help="run or resume the frozen G4 reportable matrix")
    g4_run.add_argument("index", type=Path)
    g4_run.add_argument("--workers", type=int, default=4)
    v21_g1 = sub.add_parser("v21-g1", help="run the V2.1 Stage 1 structural Monte Carlo and exact-enumeration gate")
    v21_g1.add_argument("config", type=Path)
    v21_g1.add_argument("--output-root", type=Path, default=Path("runs") / "v2_1" / "g1")
    v21_g2_build = sub.add_parser("v21-g2-build", help="freeze the 100-case/300-run V2.1 cross-layer matrix")
    v21_g2_build.add_argument("g1_run_dir", type=Path)
    v21_g2_run = sub.add_parser("v21-g2-run", help="run or resume the V2.1-G2 paired matrix")
    v21_g2_run.add_argument("index", type=Path)
    v21_g2_run.add_argument("--workers", type=int, default=4)
    v21_g2_analyze = sub.add_parser("v21-g2-analyze", help="analyze the completed V2.1-G2 paired matrix")
    v21_g2_analyze.add_argument("index", type=Path)
    v21_g3 = sub.add_parser("v21-g3", help="run the V2.1 Stage 2 weight-cloud and robust Pareto gate")
    v21_g3.add_argument("config", type=Path)
    v21_g3.add_argument("--output-root", type=Path, default=Path("runs") / "v2_1" / "g3")
    v21_g5 = sub.add_parser("v21-g5", help="run the V2.1 packet-event networked coordination gate")
    v21_g5.add_argument("config", type=Path)
    v21_g5.add_argument("--output-root", type=Path, default=Path("runs") / "v2_1" / "g5")
    v21_g6 = sub.add_parser("v21-g6", help="run the V2.1 sequential formation-recovery and mission-service gate")
    v21_g6.add_argument("config", type=Path)
    v21_g6.add_argument("--output-root", type=Path, default=Path("runs") / "v2_1" / "g6")
    v21_g7 = sub.add_parser("v21-g7", help="run the V2.1 simultaneous-failure recovery comparison")
    v21_g7.add_argument("config", type=Path)
    v21_g7.add_argument("--output-root", type=Path, default=Path("runs") / "v2_1" / "g7")
    v21_g8 = sub.add_parser("v21-g8", help="run the V2.3 online scale and path-cutoff safe-hold architecture study")
    v21_g8.add_argument("config", type=Path)
    v21_g8.add_argument("--output-root", type=Path, default=Path("runs") / "v2_1" / "g8")
    v21_g9 = sub.add_parser("v21-g9", help="run the V2.1 offline Stage 1/2 scale and recertification study")
    v21_g9.add_argument("config", type=Path)
    v21_g9.add_argument("--output-root", type=Path, default=Path("runs") / "v2_1" / "g9")
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = verify_manifest(args.run_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["valid"] else 2
    if args.command == "plot":
        outputs = generate_g1_overview(args.run_dir, show=args.show)
        print(json.dumps({"figures": [str(path) for path in outputs]}, indent=2, ensure_ascii=False))
        return 0
    if args.command in {"g2-p01", "g2-p02"}:
        runner = run_p01 if args.command == "g2-p01" else run_p02
        output = runner(args.config, args.output_root)
        print(json.dumps({"output_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "g3-opt":
        output = run_g3_solver_gate(args.policy, args.output_root)
        print(json.dumps({"output_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "g3-net-calibrate":
        output = run_g3_network_calibration(args.config, args.trace_root, args.output_root)
        print(json.dumps({"output_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "g3-net-calibrate-b2":
        output = run_g3b2_calibration(args.config, args.trace_root, args.output_root)
        print(json.dumps({"output_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "g3-net-calibrate-b3":
        output = run_g3b3_calibration(args.config, args.trace_root, args.output_root)
        print(json.dumps({"output_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "g4-build":
        output = build_g4_matrix()
        print(json.dumps({"matrix": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "g4-run":
        output = run_g4_batch(args.index, workers=args.workers)
        print(json.dumps({"progress": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v21-g1":
        output = run_v21_g1(args.config, args.output_root)
        print(json.dumps({"output_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v21-g2-build":
        output = build_v21_g2_matrix(args.g1_run_dir)
        print(json.dumps({"matrix": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v21-g2-run":
        output = run_v21_g2_batch(args.index, workers=args.workers)
        print(json.dumps({"progress": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v21-g2-analyze":
        output = analyze_v21_g2(args.index)
        print(json.dumps({"analysis_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v21-g3":
        output = run_v21_g3(args.config, args.output_root)
        print(json.dumps({"output_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v21-g5":
        output = run_v21_g5(args.config, args.output_root)
        print(json.dumps({"output_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v21-g6":
        output = run_v21_g6(args.config, args.output_root)
        print(json.dumps({"output_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v21-g7":
        output = run_v21_g7(args.config, args.output_root)
        print(json.dumps({"output_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v21-g8":
        output = run_v21_g8(args.config, args.output_root)
        print(json.dumps({"output_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "v21-g9":
        output = run_v21_g9(args.config, args.output_root)
        print(json.dumps({"output_dir": str(output)}, indent=2, ensure_ascii=False))
        return 0
    config = load_config(args.config)
    if args.command == "validate":
        print(json.dumps({
            "valid": True,
            "source": str(config.source),
            "config_hash": config.config_hash,
        }, indent=2, ensure_ascii=False))
        return 0
    run_dir, metrics = run_simulation(config)
    outputs = generate_g1_overview(run_dir, show=args.show) if (args.plot or args.show) else []
    print(json.dumps({
        "run_dir": str(run_dir),
        "figures": [str(path) for path in outputs],
        "metrics": metrics,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
