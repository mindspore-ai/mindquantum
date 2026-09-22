# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reproducible NPU end-to-end timings and complex128 correctness references.

Run as a script, without concurrent device jobs. Each timing includes the final
device synchronization. No warm-up runs are omitted. Results are JSON lines.
"""

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

import mindquantum as mq
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.simulator import Simulator
from mindquantum.utils import random_circuit


def _timed(function, *args):
    start = time.perf_counter()
    value = function(*args)
    return value, 1000 * (time.perf_counter() - start)


def _emit(kind, **data):
    print(json.dumps({'kind': kind, **data}), flush=True)


def _hamiltonian(n_qubits):
    operator = QubitOperator('', 0.137)
    for q in range(min(n_qubits, 12)):
        operator += QubitOperator(f'Z{q}', (-1) ** q * (0.2 + q / 31))
        operator += QubitOperator(f'X{q} Y{(q + 1) % n_qubits}', 0.11 + q / 47)
    return Hamiltonian(operator, mq.float32)


def _state_error(actual, reference):
    error, norm, actual_norm, maximum = 0.0, 0.0, 0.0, 0.0
    # Full-state comparison, bounded host temporary storage, double accumulation.
    for start in range(0, len(actual), 1 << 20):
        got = actual[start : start + (1 << 20)].astype(np.complex128)
        wanted = reference[start : start + (1 << 20)]
        difference = got - wanted
        error += np.vdot(difference, difference).real
        norm += np.vdot(wanted, wanted).real
        actual_norm += np.vdot(got, got).real
        maximum = max(maximum, float(np.max(np.abs(difference))))
    relative = float(np.sqrt(error / norm))
    if relative > 3e-5 or abs(actual_norm - 1) > 3e-5:
        raise AssertionError(f'Full-state accuracy failed: relative={relative}, norm={actual_norm}.')
    return {'relative_l2': relative, 'max_absolute': maximum, 'norm_error': float(abs(actual_norm - 1))}


def _evolution(n_qubits, options):
    circuit = random_circuit(n_qubits, n_qubits * 100, seed=options.seed)
    sim, construction = _timed(lambda: Simulator('mqvector_npu', n_qubits, seed=options.seed, device_id=options.device))
    _, first = _timed(sim.apply_circuit, circuit)
    actual, download = _timed(sim.get_qs)
    _emit(
        'evolution_first',
        qubits=n_qubits,
        gates=len(circuit),
        construction_ms=construction,
        first_apply_ms=first,
        first_download_ms=download,
        soc=sim.backend.sim.soc_name,
    )

    cpu = Simulator('mqvector', n_qubits, seed=options.seed, dtype=mq.complex128)
    _, cpu_ms = _timed(cpu.apply_circuit, circuit)
    reference = cpu.get_qs()
    error = _state_error(actual, reference)
    del actual, reference
    _emit('evolution_accuracy', qubits=n_qubits, cpu_complex128_ms=cpu_ms, **error)

    ham = _hamiltonian(n_qubits)
    expected = cpu.get_expectation(ham.astype(mq.float64))
    expectation_times = []
    for _ in range(options.repeats):
        value, elapsed = _timed(sim.get_expectation, ham)
        expectation_times.append(elapsed)
        np.testing.assert_allclose(value, expected, atol=3e-6, rtol=3e-5)
    _emit(
        'expectation',
        qubits=n_qubits,
        terms=len(ham.hamiltonian.terms),
        invocation_ms=expectation_times,
        absolute_error=float(abs(value - expected)),
    )

    # Explicitly distinguish resident executable replay from first-call execution.
    executable, compile_ms = _timed(sim.backend.compile, circuit)
    replay_times = []
    for _ in range(options.repeats):
        sim.reset()
        _, elapsed = _timed(sim.backend.apply_compiled, executable)
        replay_times.append(elapsed)
    _emit(
        'compiled_replay',
        qubits=n_qubits,
        prepare_ms=compile_ms,
        invocation_ms=replay_times,
        tiles=executable.tile_count,
        cube_tiles=executable.cube_tile_count,
        instructions=executable.instruction_count,
        metadata_bytes=executable.metadata_bytes,
    )


def _gradient(n_qubits, options):
    circuit = Circuit().ry(ParameterResolver('encoder').as_encoder(), 0)
    for layer in range(2):
        for q in range(n_qubits):
            circuit.ry(f'a{layer}_{q}', q).rz(0.19 + q / 43, q)
        for q in range(n_qubits):
            circuit.x((q + 1) % n_qubits, q)
    weights = np.linspace(-0.93, 0.67, 2 * n_qubits)
    encoder = np.asarray([[-0.71], [0.29], [0.83]])
    ham = _hamiltonian(n_qubits)
    sim = Simulator('mqvector_npu', n_qubits, seed=options.seed, device_id=options.device)
    op, plan_ms = _timed(sim.get_expectation_with_grad, ham, circuit)
    single, first_ms = _timed(op, encoder[:1], weights)
    batch, batch_ms = _timed(op, encoder, weights)
    repeats = []
    for _ in range(options.repeats):
        result, elapsed = _timed(op, encoder[:1], weights)
        repeats.append(elapsed)
        for value, expected in zip(result, single):
            np.testing.assert_array_equal(value, expected)
    _emit(
        'gradient_timing',
        qubits=n_qubits,
        gates=len(circuit),
        parameters=1 + len(weights),
        plan_ms=plan_ms,
        first_single_ms=first_ms,
        repeat_single_ms=repeats,
        batch_size=len(encoder),
        batch_ms=batch_ms,
    )

    cpu = Simulator('mqvector', n_qubits, seed=options.seed, dtype=mq.complex128)
    reference_op = cpu.get_expectation_with_grad(ham.astype(mq.float64), circuit)
    reference, cpu_ms = _timed(reference_op, encoder, weights)
    errors = []
    for value, expected in zip(batch, reference):
        errors.append(float(np.max(np.abs(value - expected))))
        np.testing.assert_allclose(value, expected, atol=2e-5, rtol=3e-5)
    _emit(
        'gradient_accuracy',
        qubits=n_qubits,
        cpu_complex128_batch_ms=cpu_ms,
        expectation_max_error=errors[0],
        encoder_max_error=errors[1],
        ansatz_max_error=errors[2],
    )


def main():
    """Benchmark the public API, then independently check its outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qubits', nargs='+', type=int, default=[24, 28])
    parser.add_argument('--gradient-qubits', type=int, default=20)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=831)
    parser.add_argument('--repeats', type=int, default=3)
    options = parser.parse_args()
    if options.repeats < 1 or min(*options.qubits, options.gradient_qubits) < 2:
        parser.error('Use positive repeats and at least two qubits.')
    _emit(
        'environment',
        python=platform.python_version(),
        host=platform.machine(),
        numpy=np.__version__,
        mindquantum=(Path(__file__).resolve().parents[1] / 'VERSION.txt').read_text().strip(),
        seed=options.seed,
        device=options.device,
        cann=os.getenv('ASCEND_HOME_PATH'),
        omp_threads=os.getenv('OMP_NUM_THREADS'),
        openblas_threads=os.getenv('OPENBLAS_NUM_THREADS'),
    )
    for n_qubits in options.qubits:
        _evolution(n_qubits, options)
        gc.collect()
    _gradient(options.gradient_qubits, options)


if __name__ == '__main__':
    main()
