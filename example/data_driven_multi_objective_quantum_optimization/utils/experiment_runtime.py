"""Shared runtime validation and reproducibility metadata for experiment runners."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

PACKAGE_DISTRIBUTIONS = (
    "torch",
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "optuna",
    "dimod",
    "dwave-neal",
)
TRAINING_REQUIRED_MODULES = ("torch", "optuna", "numpy", "scipy", "sklearn", "neal", "dimod")

NEAL_SEED_MAX = (2**31) - 1
NUMPY_SEED_MAX = (2**32) - 1
FM_SEED_MIN = NEAL_SEED_MAX + 1
SEED_DERIVATION_SCHEME = "mixed_radix_v1"
FIGURE4_SEED_INDEX = 0
FIGURE5_SEED_INDEX = 1
FIGURE4_INITIAL_NAMESPACE = 1
FIGURE5_INITIAL_NAMESPACE = 2
FIGURE4_REPLACEMENT_NAMESPACE = 3
FIGURE5_REPLACEMENT_NAMESPACE = 4
FIGURE5_PREFERENCE_NAMESPACE = 5
FM_TRAINING_BACKEND = "pytorch_fm_lbfgs"
FM_FACTORIZATION_RANK = 6
MAX_RANDOM_REPLACEMENT_ATTEMPTS = 10_000
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def require_integer(
    name: str,
    value: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return a Python ``int`` without accepting lossy or bool coercions."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return normalized


def fm_seed_block_size(optuna_trials: int) -> int:
    """Return the fixed-width seed block reserved for one FM fit."""
    return require_integer("optuna_trials", optuna_trials, minimum=0) + 4


def fm_seed_plan(seed_root: int, optuna_trials: int) -> dict[str, Any]:
    """Allocate split, sampler, trial, and final-fit seeds inside one block."""
    trial_count = require_integer("optuna_trials", optuna_trials, minimum=0)
    block_size = fm_seed_block_size(trial_count)
    root = require_integer(
        "seed",
        seed_root,
        minimum=0,
        maximum=NUMPY_SEED_MAX - block_size + 1,
    )
    return {
        "root": root,
        "block_size": block_size,
        "split": [root, root + 1],
        "tuner": root + 2,
        "trials": list(range(root + 3, root + 3 + trial_count)),
        "final_fit": root + 3 + trial_count,
    }


def ensure_training_dependencies(required_modules: Sequence[str]) -> None:
    """Fail before output creation when training dependencies are unavailable."""
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(f"Missing required dependencies: {', '.join(missing)}")


def ensure_compute_device_available(device: str) -> None:
    """Fail before an experiment starts when its requested device is unavailable."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise OSError("Requested --device cuda, but torch.cuda.is_available() is false.")


def resolve_contiguous_seeds(default_count: int, requested_count: int | None, seed_start: int) -> list[int]:
    """Build a validated non-negative contiguous seed list."""
    count = require_integer(
        "num_seeds",
        default_count if requested_count is None else requested_count,
    )
    start = require_integer("seed_start", seed_start, minimum=0, maximum=NEAL_SEED_MAX)
    if count <= 0:
        raise ValueError("num_seeds must be positive")
    final_seed = start + count - 1
    if final_seed > NEAL_SEED_MAX:
        raise ValueError(f"contiguous seed range ends at {final_seed}, above the neal maximum {NEAL_SEED_MAX}")
    return validate_seed_list(range(start, start + count))


def validate_seed_list(
    seed_list: Sequence[int],
    *,
    iterations: int | None = None,
) -> list[int]:
    """Normalize and validate base seeds.

    ``iterations`` is retained as a compatibility keyword for callers that resolve a
    configuration before validating seeds. Derived schedules are checked separately by
    :func:`validate_seed_schedule`.
    """
    if len(seed_list) == 0:
        raise ValueError("seed_list must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seed_list):
        raise ValueError("seed_list must contain integers")
    normalized = [int(seed) for seed in seed_list]
    if any(seed < 0 for seed in normalized):
        raise ValueError("seeds must be non-negative")
    if any(seed > NEAL_SEED_MAX for seed in normalized):
        raise ValueError(f"seeds must not exceed the base-seed maximum {NEAL_SEED_MAX}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("seed_list must not contain duplicates")
    if iterations is not None:
        require_integer("iterations", iterations, minimum=1)
    return normalized


def derive_bounded_seed(
    base_seed: int,
    *,
    trajectory_count: int,
    trajectory_index: int,
    iterations: int,
    iteration: int,
    stream_count: int,
    stream_index: int,
    figure_index: int,
) -> int:
    """Derive a dependency-bounded seed using the ``mixed_radix_v1`` layout."""
    base = require_integer("base_seed", base_seed, minimum=0, maximum=NEAL_SEED_MAX)
    trajectory_total = require_integer("trajectory_count", trajectory_count, minimum=1)
    trajectory = require_integer("trajectory_index", trajectory_index, minimum=0)
    iteration_total = require_integer("iterations", iterations, minimum=1)
    iteration_value = require_integer("iteration", iteration, minimum=0)
    stream_total = require_integer("stream_count", stream_count, minimum=1)
    stream = require_integer("stream_index", stream_index, minimum=0)
    figure = require_integer("figure_index", figure_index, minimum=0, maximum=1)
    if trajectory >= trajectory_total:
        raise ValueError("trajectory_index must be less than trajectory_count")
    if iteration_value >= iteration_total:
        raise ValueError("iteration must be less than iterations")
    if stream >= stream_total:
        raise ValueError("stream_index must be less than stream_count")
    inner = ((base * trajectory_total + trajectory) * iteration_total + iteration_value) * stream_total + stream
    derived = (inner * 2) + figure
    if derived > NEAL_SEED_MAX:
        raise ValueError(f"derived bounded seed {derived} exceeds dependency maximum {NEAL_SEED_MAX}")
    return derived


def derive_fm_seed_root(
    base_seed: int,
    *,
    trajectory_count: int,
    trajectory_index: int,
    iterations: int,
    iteration: int,
    model_count: int,
    model_index: int,
    figure_index: int,
    block_size: int,
) -> int:
    """Return the first seed of one non-overlapping FM training block."""
    base = require_integer("base_seed", base_seed, minimum=0, maximum=NEAL_SEED_MAX)
    trajectory_total = require_integer("trajectory_count", trajectory_count, minimum=1)
    trajectory = require_integer("trajectory_index", trajectory_index, minimum=0)
    iteration_total = require_integer("iterations", iterations, minimum=1)
    iteration_value = require_integer("iteration", iteration, minimum=0)
    models = require_integer("model_count", model_count, minimum=1)
    model = require_integer("model_index", model_index, minimum=0)
    figure = require_integer("figure_index", figure_index, minimum=0, maximum=1)
    block = require_integer("block_size", block_size, minimum=4)
    if trajectory >= trajectory_total:
        raise ValueError("trajectory_index must be less than trajectory_count")
    if iteration_value >= iteration_total:
        raise ValueError("iteration must be less than iterations")
    if model >= models:
        raise ValueError("model_index must be less than model_count")
    call_index = (
        ((base * trajectory_total + trajectory) * iteration_total + iteration_value) * models + model
    ) * 2 + figure
    root = FM_SEED_MIN + (call_index * block)
    final_seed = root + block - 1
    if final_seed > NUMPY_SEED_MAX:
        raise ValueError(f"derived FM seed block ends at {final_seed}, above the NumPy maximum {NUMPY_SEED_MAX}")
    return root


def derive_python_seed(
    namespace_id: int,
    base_seed: int,
    *,
    trajectory_index: int = 0,
    iteration: int = 0,
) -> int:
    """Pack stable fields into an unbounded seed for ``random.Random``."""
    namespace = require_integer("namespace_id", namespace_id, minimum=0, maximum=(2**64) - 1)
    base = require_integer("base_seed", base_seed, minimum=0, maximum=(2**64) - 1)
    trajectory = require_integer("trajectory_index", trajectory_index, minimum=0, maximum=(2**64) - 1)
    iteration_value = require_integer("iteration", iteration, minimum=0, maximum=(2**64) - 1)
    return (namespace << 192) | (base << 128) | (trajectory << 64) | iteration_value


def validate_seed_schedule(
    seed_list: Sequence[int],
    *,
    figure_index: int,
    trajectory_count: int,
    iterations: int,
    bounded_stream_count: int,
    fm_model_count: int,
    fm_seed_block_size: int,
) -> list[int]:
    """Validate the maximum derived bounded and FM seeds before any run output exists."""
    seeds = validate_seed_list(seed_list)
    trajectory_total = require_integer("trajectory_count", trajectory_count, minimum=1)
    iteration_total = require_integer("iterations", iterations, minimum=1)
    streams = require_integer("bounded_stream_count", bounded_stream_count, minimum=1)
    models = require_integer("fm_model_count", fm_model_count, minimum=1)
    block = require_integer("fm_seed_block_size", fm_seed_block_size, minimum=4)
    maximum_base = max(seeds)
    derive_bounded_seed(
        maximum_base,
        trajectory_count=trajectory_total,
        trajectory_index=trajectory_total - 1,
        iterations=iteration_total,
        iteration=iteration_total - 1,
        stream_count=streams,
        stream_index=streams - 1,
        figure_index=figure_index,
    )
    derive_fm_seed_root(
        maximum_base,
        trajectory_count=trajectory_total,
        trajectory_index=trajectory_total - 1,
        iterations=iteration_total,
        iteration=iteration_total - 1,
        model_count=models,
        model_index=models - 1,
        figure_index=figure_index,
        block_size=block,
    )
    return seeds


def same_json_value(actual: Any, expected: Any) -> bool:
    """Compare JSON values without treating bool and int as interchangeable."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_json_value(actual[key], expected[key]) for key in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_json_value(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def load_json_object(path: str | Path, *, document_name: str = "JSON document") -> dict[str, Any]:
    """Load one JSON object and wrap all read/parse errors with path context."""
    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {document_name} {input_path}: cannot read valid JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid {document_name} {input_path}: root must be a JSON object")
    return payload


def write_json_atomic(
    path: str | Path,
    payload: dict[str, Any],
    *,
    indent: int | None = 2,
) -> Path:
    """Stream JSON to a unique same-directory file, fsync it, then atomically replace."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=indent,
                separators=(",", ":") if indent is None else None,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return output_path


def configure_file_logging_path(log_path: str | Path) -> Path:
    """Configure the process-wide experiment file handler."""
    output_path = Path(log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        if getattr(handler, "_experiment_file_handler", False):
            root_logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(output_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler._experiment_file_handler = True  # type: ignore[attr-defined]
    root_logger.addHandler(file_handler)
    return output_path


def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(workspace_root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(workspace_root),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def collect_runtime_metadata(
    workspace_root: str | Path,
    reference_paper: str | Path | None = None,
) -> dict[str, Any]:
    """Collect enough environment/source identity to audit a generated result."""
    root = Path(workspace_root).resolve()
    package_versions: dict[str, str | None] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None

    paper_metadata: dict[str, Any] | None = None
    if reference_paper is not None:
        paper_path = Path(reference_paper)
        if not paper_path.is_absolute():
            paper_path = root / paper_path
        paper_metadata = {
            "path": str(paper_path),
            "exists": paper_path.is_file(),
            "sha256": sha256_file(paper_path) if paper_path.is_file() else None,
        }

    import torch

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": package_versions,
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "git": _git_metadata(root),
        "reference_paper": paper_metadata,
    }


__all__ = [
    "FIGURE4_INITIAL_NAMESPACE",
    "FIGURE4_REPLACEMENT_NAMESPACE",
    "FIGURE4_SEED_INDEX",
    "FIGURE5_INITIAL_NAMESPACE",
    "FIGURE5_PREFERENCE_NAMESPACE",
    "FIGURE5_REPLACEMENT_NAMESPACE",
    "FIGURE5_SEED_INDEX",
    "FM_SEED_MIN",
    "FM_FACTORIZATION_RANK",
    "FM_TRAINING_BACKEND",
    "MAX_RANDOM_REPLACEMENT_ATTEMPTS",
    "NEAL_SEED_MAX",
    "NUMPY_SEED_MAX",
    "SEED_DERIVATION_SCHEME",
    "TRAINING_REQUIRED_MODULES",
    "collect_runtime_metadata",
    "configure_file_logging_path",
    "derive_bounded_seed",
    "derive_fm_seed_root",
    "derive_python_seed",
    "ensure_compute_device_available",
    "ensure_training_dependencies",
    "fm_seed_block_size",
    "fm_seed_plan",
    "load_json_object",
    "require_integer",
    "resolve_contiguous_seeds",
    "same_json_value",
    "sha256_file",
    "validate_seed_list",
    "validate_seed_schedule",
    "write_json_atomic",
]
