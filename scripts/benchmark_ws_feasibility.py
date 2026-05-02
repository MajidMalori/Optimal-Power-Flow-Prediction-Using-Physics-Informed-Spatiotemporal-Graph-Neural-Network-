#!/usr/bin/env python3
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.benchmarks.benchmark_dataset import load_states_jsonl
from src.benchmarks.feasibility_runner import run_feasibility_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start feasibility benchmark")
    parser.add_argument("--case", type=str, required=True, help="case33/case57/case118")
    parser.add_argument("--max-samples", type=int, default=10)
    args = parser.parse_args()
    from src.benchmarks.feasibility_runtime import run_all_methods_for_state_feasibility
    from src.visualization.plot_ws_feasibility import plot_ws_feasibility

    case = args.case if args.case.startswith("case") else f"case{args.case}"
    states_path = os.path.join(PROJECT_ROOT, "data", "benchmark", case, "states.jsonl")
    states = load_states_jsonl(states_path)[: args.max_samples]

    records, summary = run_feasibility_benchmark(states, run_all_methods_for_state_feasibility)

    out_dir = os.path.join(PROJECT_ROOT, "reports", "warmstart", "feasibility", case)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{case}_feasibility_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, f"{case}_feasibility_records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    plot_ws_feasibility(records, summary, case_name=case, output_dir=out_dir)

    print(f"[{case}] feasibility summary written to {out_dir}")


if __name__ == "__main__":
    main()
