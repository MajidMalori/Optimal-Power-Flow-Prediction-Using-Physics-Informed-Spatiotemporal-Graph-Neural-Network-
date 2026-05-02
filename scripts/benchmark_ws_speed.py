#!/usr/bin/env python3
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.benchmarks.benchmark_dataset import load_states_jsonl
from src.benchmarks.speed_runner import run_speed_benchmark
from src.benchmarks.speed_runtime import run_all_methods_for_state


def main() -> None:
    from src.visualization.plot_ws_speed import plot_ws_speed
    parser = argparse.ArgumentParser(description="Warm-start speed benchmark")
    parser.add_argument("--case", type=str, required=True, help="case33/case57/case118")
    parser.add_argument("--max-samples", type=int, default=10)
    args = parser.parse_args()

    case = args.case if args.case.startswith("case") else f"case{args.case}"
    states_path = os.path.join(PROJECT_ROOT, "data", "benchmark", case, "states.jsonl")
    states = load_states_jsonl(states_path)[: args.max_samples]

    records, summary = run_speed_benchmark(states, run_all_methods_for_state)

    out_dir = os.path.join(PROJECT_ROOT, "reports", "warmstart", "speed", case)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{case}_speed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, f"{case}_speed_records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    plot_ws_speed(records, summary, case_name=case, output_dir=out_dir)

    print(f"[{case}] speed summary written to {out_dir}")


if __name__ == "__main__":
    main()
