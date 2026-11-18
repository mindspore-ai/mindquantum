# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Process-level guard against native heap corruption in the ``mqmatrix`` backend.

MindQuantum 0.13 rewrote ``mqmatrix`` on top of the shared circuit-tiling engine
and changed the OpenMP thread policy to an adaptive one.  On the reference
Windows build the stock density-matrix system tests abort the interpreter with
``STATUS_HEAP_CORRUPTION`` (``0xC0000374`` / exit code ``-1073740940``) when more
than one OpenMP thread is allowed, while ``OMP_NUM_THREADS=1`` makes the same
tests pass.

A native heap corruption cannot be caught from Python: the interpreter dies
before any exception can be raised, and CPython's own ``faulthandler`` only
prints the native stack.  The only reliable way to observe it from a test is to
run the workload in an isolated interpreter and inspect the exit code, which is
what this module does.

The test is intentionally independent of the outer ``OMP_NUM_THREADS`` setting:
it always requests the host parallelism that reproduced the crash, so running
the suite through a single-threaded wrapper cannot hide the defect.

While the defect is open, the multi-threaded guard is marked ``xfail``
(``strict=False``) so that the level-0 gate stays green; it becomes a hard
failure as soon as the marker is removed together with the fix.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]

_TILING_TESTS = Path(__file__).resolve().parent / 'test_mqmatrix_circuit_tiling.py'
_REPETITIONS = 3
_INNER_TIMEOUT = 600

# Windows NTSTATUS values that indicate a native memory defect rather than a test failure.
_NATIVE_CRASHES = {
    0xC0000374: 'STATUS_HEAP_CORRUPTION',
    0xC0000005: 'STATUS_ACCESS_VIOLATION',
    0xC00000FD: 'STATUS_STACK_OVERFLOW',
    0xC0000409: 'STATUS_STACK_BUFFER_OVERRUN',
    0xC0000602: 'STATUS_FAIL_FAST_EXCEPTION',
}

_HEAP_CORRUPTION = pytest.mark.xfail(
    reason='Open defect: mqmatrix aborts with STATUS_HEAP_CORRUPTION (0xC0000374) under multi-threaded OpenMP.',
    strict=False,
)


def _describe_exit_code(returncode):
    """Return a human readable description of a process exit code."""
    unsigned = returncode & 0xFFFFFFFF
    name = _NATIVE_CRASHES.get(unsigned)
    if name is not None:
        return f'exit code {returncode} ({unsigned:#010x}, {name})'
    if returncode >= 0:
        return f'exit code {returncode} (tests failed inside the child process)'
    return f'exit code {returncode} ({unsigned:#010x}, native abort)'


def _tail(text, lines=12):
    """Return the last lines of a captured stream."""
    return '\n'.join(text.splitlines()[-lines:])


@_HEAP_CORRUPTION
def test_multithreaded_density_matrix_suite_does_not_abort():
    """
    Feature: Thread safety of the rewritten density matrix backend.

    Description: Run the ``mqmatrix`` circuit-tiling system tests in an isolated interpreter,
        three times, with the host OpenMP parallelism that reproduces the native crash.

    Expectation: Every child interpreter exits with code 0.  A negative exit code such as
        ``0xC0000374`` proves that the native backend corrupted the heap and the process was
        terminated by the operating system instead of raising a Python exception.
    """
    if not _TILING_TESTS.exists():
        pytest.skip(f'{_TILING_TESTS} is missing')

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(max(2, os.cpu_count() or 2))
    env.pop('MQ_LOG', None)
    command = [
        sys.executable,
        '-m',
        'pytest',
        str(_TILING_TESTS),
        '-q',
        '-p',
        'no:cacheprovider',
    ]

    failures = []
    for attempt in range(1, _REPETITIONS + 1):
        try:
            completed = subprocess.run(
                command,
                cwd=str(_TILING_TESTS.parent),
                env=env,
                capture_output=True,
                text=True,
                timeout=_INNER_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            failures.append(f'attempt {attempt}: the child process did not finish within {_INNER_TIMEOUT} s')
            continue
        if completed.returncode != 0:
            failures.append(
                f'attempt {attempt}: {_describe_exit_code(completed.returncode)}\n'
                f'--- child output (tail) ---\n{_tail(completed.stdout + completed.stderr)}'
            )

    assert not failures, (
        'The mqmatrix backend aborted under multi-threaded OpenMP '
        f'(OMP_NUM_THREADS={env["OMP_NUM_THREADS"]}); '
        f'{len(failures)}/{_REPETITIONS} repetitions failed.\n' + '\n\n'.join(failures)
    )


def test_single_threaded_density_matrix_suite_is_stable():
    """
    Feature: Thread dependence of the density matrix crash.

    Description: Run the same circuit-tiling system tests with ``OMP_NUM_THREADS=1``.

    Expectation: The child interpreter exits with code 0, which isolates the defect to the
        parallel execution path and documents the supported workaround.
    """
    if not _TILING_TESTS.exists():
        pytest.skip(f'{_TILING_TESTS} is missing')

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env.pop('MQ_LOG', None)
    completed = subprocess.run(
        [sys.executable, '-m', 'pytest', str(_TILING_TESTS), '-q', '-p', 'no:cacheprovider'],
        cwd=str(_TILING_TESTS.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=_INNER_TIMEOUT,
        check=False,
    )
    assert completed.returncode == 0, (
        f'OMP_NUM_THREADS=1 must be a valid workaround, but the child process returned '
        f'{_describe_exit_code(completed.returncode)}\n{_tail(completed.stdout + completed.stderr)}'
    )
