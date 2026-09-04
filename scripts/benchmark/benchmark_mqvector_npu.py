#!/usr/bin/env python3
"""Reproducible single-card CPU/NPU benchmark for the mqvector_npu backend.

This script intentionally does not run as part of the normal test suite. Run it
on a machine with CANN, an available NPU, and the ``mqvector_npu`` extension
built.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import mindquantum as mq
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import CNOT
from mindquantum.simulator import Simulator
from mindquantum.utils import random_circuit


GATE_NAMES = ("X", "Y", "Z", "H", "RX", "RY", "RZ", "CNOT", "CZ")


@dataclass(frozen=True)
class BenchmarkRow:
    """One timing result for a backend and benchmark workload."""

    group: str
    case: str
    backend: str
    n_qubits: int
    depth: int
    seed: int
    warmup: int
    runs: int
    apply_ms: list[float]
    read_ms: list[float]


def _apply_named_gate(circuit: Circuit, gate_name: str, layer: int, n_qubits: int, rng) -> None:
    """Append one named gate to a benchmark circuit."""

    target = layer % n_qubits
    if gate_name == "X":
        circuit.x(target)
    elif gate_name == "Y":
        circuit.y(target)
    elif gate_name == "Z":
        circuit.z(target)
    elif gate_name == "H":
        circuit.h(target)
    elif gate_name == "RX":
        circuit.rx(float(rng.uniform(-np.pi, np.pi)), target)
    elif gate_name == "RY":
        circuit.ry(float(rng.uniform(-np.pi, np.pi)), target)
    elif gate_name == "RZ":
        circuit.rz(float(rng.uniform(-np.pi, np.pi)), target)
    elif gate_name == "CNOT":
        circuit.append(CNOT.on((target + 1) % n_qubits, target))
    elif gate_name == "CZ":
        circuit.z((target + 1) % n_qubits, target)
    else:
        raise ValueError(f"unsupported benchmark gate: {gate_name}")


def make_gate_circuit(gate_name: str, n_qubits: int, depth: int, seed: int) -> Circuit:
    rng = np.random.default_rng(seed)
    circuit = Circuit()
    for layer in range(depth):
        _apply_named_gate(circuit, gate_name, layer, n_qubits, rng)
    return circuit


def make_mixed_circuit(n_qubits: int, depth: int, seed: int) -> Circuit:
    rng = np.random.default_rng(seed)
    circuit = Circuit()
    for layer in range(depth):
        _apply_named_gate(circuit, GATE_NAMES[layer % len(GATE_NAMES)], layer, n_qubits, rng)
    return circuit


def make_random_circuit(n_qubits: int, gate_num: int, sd_rate: float, ctrl_rate: float, seed: int) -> Circuit:
    """Build the acceptance workload through MindQuantum's public random_circuit API."""
    return random_circuit(n_qubits, gate_num, sd_rate, ctrl_rate, seed=seed)


def run_case(
    group: str,
    case: str,
    backend: str,
    n_qubits: int,
    depth: int,
    seed: int,
    circuit: Circuit,
    warmup: int,
    runs: int,
) -> BenchmarkRow:
    """Measure circuit application and state readback for one backend."""

    if backend == "mqvector":
        simulator = Simulator(backend, n_qubits, dtype=mq.complex64)
    else:
        simulator = Simulator(backend, n_qubits)

    for _ in range(warmup):
        simulator.reset()
        simulator.apply_circuit(circuit)
        simulator.get_qs()

    apply_ms = []
    read_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        simulator.reset()
        simulator.apply_circuit(circuit)
        after_apply = time.perf_counter()
        simulator.get_qs()
        after_read = time.perf_counter()
        apply_ms.append((after_apply - start) * 1000.0)
        read_ms.append((after_read - after_apply) * 1000.0)

    return BenchmarkRow(group, case, backend, n_qubits, depth, seed, warmup, runs, apply_ms, read_ms)


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or any(
        item <= 0 for item in values
    ):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-qubits", type=parse_int_list, default=(16, 20, 24))
    parser.add_argument("--large-depth", type=int, default=32)
    parser.add_argument("--random-qubits", type=parse_int_list, default=(20, 24, 26))
    parser.add_argument("--random-gates-per-qubit", type=int, default=100)
    parser.add_argument("--random-sd-rate", type=float, default=0.5)
    parser.add_argument("--random-ctrl-rate", type=float, default=0.2)
    parser.add_argument(
        "--acceptance-random",
        action="store_true",
        help="also run random_circuit(n, n*100) for the mentor acceptance workload",
    )
    parser.add_argument(
        "--acceptance-only",
        action="store_true",
        help="run only the mentor acceptance random-circuit workload",
    )
    parser.add_argument("--gate-qubits", type=int, default=8)
    parser.add_argument("--gate-depth", type=int, default=32)
    parser.add_argument("--depths", type=parse_int_list, default=(8, 16, 32, 64))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output", type=Path)
    return parser


def _validate_args(args, parser: argparse.ArgumentParser) -> None:
    """Validate benchmark dimensions, rates, and repetition counts."""
    if (
        args.large_depth <= 0
        or args.random_gates_per_qubit <= 0
        or args.gate_qubits < 2
        or args.gate_depth <= 0
        or args.warmup < 0
        or args.runs <= 0
        or any(n_qubits < 2 for n_qubits in args.large_qubits)
        or any(n_qubits < 2 for n_qubits in args.random_qubits)
        or not 0.0 <= args.random_sd_rate <= 1.0
        or not 0.0 <= args.random_ctrl_rate <= 1.0
    ):
        parser.error(
            "qubit counts must be at least 2; depths, gate counts, and runs must be positive; "
            "rates must be in [0, 1]; warmup cannot be negative"
        )


def _run_qubit_scale(args) -> list[BenchmarkRow]:
    """Run the mixed-gate workloads used for qubit scaling."""
    rows = []
    for backend in ("mqvector", "mqvector_npu"):
        for index, n_qubits in enumerate(args.large_qubits):
            circuit = make_mixed_circuit(n_qubits, args.large_depth, args.seed + index)
            rows.append(
                run_case(
                    "qubit_scale",
                    str(n_qubits),
                    backend,
                    n_qubits,
                    args.large_depth,
                    args.seed + index,
                    circuit,
                    args.warmup,
                    args.runs,
                )
            )
    return rows


def _run_acceptance_random(args) -> list[BenchmarkRow]:
    """Run the mentor acceptance random-circuit workloads."""
    rows = []
    for index, n_qubits in enumerate(args.random_qubits):
        gate_num = n_qubits * args.random_gates_per_qubit
        circuit = make_random_circuit(
            n_qubits,
            gate_num,
            args.random_sd_rate,
            args.random_ctrl_rate,
            args.seed + 300 + index,
        )
        for backend in ("mqvector", "mqvector_npu"):
            rows.append(
                run_case(
                    "random_circuit",
                    str(n_qubits),
                    backend,
                    n_qubits,
                    gate_num,
                    args.seed + 300 + index,
                    circuit,
                    args.warmup,
                    args.runs,
                )
            )
    return rows


def _run_gate_types(args) -> list[BenchmarkRow]:
    """Run one workload for each supported benchmark gate type."""
    rows = []
    for index, gate_name in enumerate(GATE_NAMES):
        circuit = make_gate_circuit(gate_name, args.gate_qubits, args.gate_depth, args.seed + 100 + index)
        for backend in ("mqvector", "mqvector_npu"):
            rows.append(
                run_case(
                    "gate_type",
                    gate_name,
                    backend,
                    args.gate_qubits,
                    args.gate_depth,
                    args.seed + 100 + index,
                    circuit,
                    args.warmup,
                    args.runs,
                )
            )
    return rows


def _run_depths(args) -> list[BenchmarkRow]:
    """Run mixed-gate workloads across the requested circuit depths."""
    rows = []
    for index, depth in enumerate(args.depths):
        circuit = make_mixed_circuit(args.gate_qubits, depth, args.seed + 200 + index)
        for backend in ("mqvector", "mqvector_npu"):
            rows.append(
                run_case(
                    "depth",
                    str(depth),
                    backend,
                    args.gate_qubits,
                    depth,
                    args.seed + 200 + index,
                    circuit,
                    args.warmup,
                    args.runs,
                )
            )
    return rows


def _build_result(args, rows: list[BenchmarkRow]) -> dict:
    """Build the JSON-serializable benchmark result."""
    environment_names = (
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_TOOLKIT_HOME_PATH",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MQVECTOR_NPU_FUSE_CIRCUIT",
        "MQVECTOR_NPU_LOCAL_FUSION",
        "MQVECTOR_NPU_LEGACY_BLOCK_DIM",
    )
    return {
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cpu_dtype": "complex64",
            "npu_dtype": "complex64",
            "single_card": True,
            "distributed": False,
            "environment": {name: os.environ.get(name) for name in environment_names},
        },
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "rows": [asdict(row) for row in rows],
    }


def _write_result(args, result: dict) -> None:
    """Write a benchmark result to a file or standard output."""
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def main() -> None:
    """Run the requested CPU/NPU benchmark groups."""
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    rows = []
    if not args.acceptance_only:
        rows.extend(_run_qubit_scale(args))

    if args.acceptance_random or args.acceptance_only:
        rows.extend(_run_acceptance_random(args))

    if not args.acceptance_only:
        rows.extend(_run_gate_types(args))
        rows.extend(_run_depths(args))

    _write_result(args, _build_result(args, rows))


if __name__ == "__main__":
    main()
