import json
import os
from dataclasses import asdict
from typing import List

from src.benchmarks.benchmark_state import BenchmarkState, validate_state


def save_states_jsonl(states: List[BenchmarkState], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for state in states:
            validate_state(state)
            f.write(json.dumps(asdict(state)) + "\n")


def load_states_jsonl(input_path: str) -> List[BenchmarkState]:
    out: List[BenchmarkState] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            state = BenchmarkState(**raw)
            validate_state(state)
            out.append(state)
    return out
