"""CPU-only data support for unconstrained, all-binary QPLIB instances.

The files served by https://qplib.zib.de/qplib/ use the LOWER TRIANGULAR
matrix T of https://qplib.zib.de/doc.html, with objective

    f(z) = 0.5 * sum(T[i,j] * z[i] * z[j], i >= j) + b @ z + c.

Do not reflect T unchanged into a symmetric Hessian: its off-diagonal entries
must first be halved. The website's convention differs from the symmetric-Q
notation in the QPLIB paper. For example, QPLIB_3506.qplib has "2 1 4.0",
whereas https://qplib.zib.de/mod/QPLIB_3506.mod has the term +2*b2*b3.
The symmetric Hessian is H = (T + T.T) / 2, including H[i,i] = T[i,i].

Record order, comments, D/E exponents and implicit binary bounds follow
Appendix B, Table 8 of Furini et al., "QPLIB: A Library of Quadratic Programming
Instances", https://doi.org/10.1007/s12532-018-0147-4. Data is licensed CC-BY 4.0
(https://creativecommons.org/licenses/by/4.0/); please attribute QPLIB and cite
that paper when publishing benchmark results. Downloaded files are unmodified.

Usage: python qplib.py download [3506 3565 ...] --directory data
With no IDs, download the repository's 23 benchmarks. Existing files are fully
validated and reused, never overwritten. Downloads have no retries, redirects,
or mirror fallback, and are published atomically only after validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.client import HTTPException
import math
from numbers import Integral
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.request import HTTPRedirectHandler, build_opener

import numpy as np


DEFAULT_INSTANCE_IDS = (
    3506, 3565, 3642, 3650, 3693, 3705, 3706, 3738, 3745, 3822, 3832, 3838,
    3850, 3852, 3877, 5721, 5725, 5755, 5875, 5881, 5882, 5909, 5922,
)
BASE_URL = "https://qplib.zib.de/qplib/"


class QplibFormatError(ValueError):
    """Malformed QPLIB data, with source and record context."""


class UnsupportedQplibError(QplibFormatError):
    """A problem outside the unconstrained, all-binary subset."""


@dataclass(frozen=True)
class IsingMetadata:
    """Affine mapping: f(z) = offset + sense_sign * scale * E(s).

    sense_sign is +1 for minimize, -1 for maximize; scale is strictly positive.
    The identity is exact before the solver coefficients are rounded to float32.
    Use QplibProblem.objective, not this mapping, to report benchmark objectives.
    """

    scale: float
    offset: float
    sense_sign: int

    def to_objective(self, energy):
        """Map normalized Ising energies back to original objective units."""
        return self.offset + self.sense_sign * self.scale * np.asarray(energy)


@dataclass(frozen=True)
class QplibProblem:
    """Original objective, with zero-based lower-triangle coordinate arrays.

    q_value contains T's literal file values (NOT a mirrored Hessian).
    linear is the fully expanded default/override vector. All coefficient
    arrays are float64, except the integer indices, and are read-only when
    obtained from parse_qplib/load_qplib. No variables, including isolated ones,
    are dropped. sense is 'minimize' or 'maximize'.
    """

    name: str
    problem_type: str
    sense: str
    n: int
    q_row: np.ndarray
    q_col: np.ndarray
    q_value: np.ndarray
    linear: np.ndarray
    constant: float
    infinity: float

    def objective(self, binary: np.ndarray) -> np.ndarray:
        """Evaluate ORIGINAL f on a binary N x B matrix, returning float64 (B,).

        This uses the original sparse polynomial, independently of to_ising.
        Spin or continuous solver states are rejected, not silently thresholded.
        """
        z = np.asarray(binary)
        if z.ndim != 2 or z.shape[0] != self.n:
            raise ValueError(f"binary must have shape ({self.n}, B)")
        if z.dtype.kind not in "biuf" or not np.all((z == 0) | (z == 1)):
            raise ValueError("binary must contain only numeric 0 and 1 values")
        z = z.astype(np.float64, copy=False)
        value = self.linear @ z + self.constant
        # Bound temporary memory independently of the number of quadratic terms.
        for start in range(0, self.q_value.size, 4096):
            part = slice(start, start + 4096)
            products = z[self.q_row[part]] * z[self.q_col[part]]
            value += (0.5 * self.q_value[part]) @ products
        return value

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, IsingMetadata]:
        """Return normalized C-contiguous float32 J (N,N), h (N,1), metadata.

        Solver convention: minimize E(s) = -0.5*s.T@J@s - h.T@s, z=(s+1)/2.
        Put a_i=b_i+T_ii/2 and w_ij=T_ij/2 (i>j). Then

          f = k + sum(w_ij*s_i*s_j/4) + sum(g_i*s_i),
          g_i = a_i/2 + sum(j != i, w_ij)/4,
          k = c + sum(a_i)/2 + sum(i>j, w_ij)/4.

        For t=+1 (min) or -1 (max), J_ij=-t*w_ij/4, h_i=-t*g_i.
        J_ii=0: binary diagonal terms contribute to g and k, not self-coupling.
        Divide J and h by max(abs(J), abs(h)), or 1 for a constant objective.
        The metadata offset is k in ORIGINAL, not sense-adjusted, units.
        """
        sign = {"minimize": 1, "maximize": -1}[self.sense]
        diagonal = self.q_row == self.q_col
        row, col = self.q_row[~diagonal], self.q_col[~diagonal]
        pair = self.q_value[~diagonal] / 8.0
        with np.errstate(over="raise", invalid="raise"):
            try:
                field = self.linear / 2.0
                np.add.at(field, self.q_row[diagonal], self.q_value[diagonal] / 4.0)
                np.add.at(field, row, pair)
                np.add.at(field, col, pair)
                offset = float(
                    self.constant + (self.linear / 2.0).sum()
                    + (self.q_value[diagonal] / 4.0).sum() + pair.sum()
                )
            except FloatingPointError as exc:
                raise ValueError("Ising conversion overflows float64") from exc
        if not math.isfinite(offset) or not np.all(np.isfinite(field)):
            raise ValueError("Ising conversion overflows float64")
        scale = max(float(np.max(np.abs(pair), initial=0.0)), float(np.abs(field).max()))
        scale = scale or 1.0
        J = np.zeros((self.n, self.n), dtype=np.float32)
        J[row, col] = -sign * (pair / scale)
        J[col, row] = J[row, col]
        h = np.asarray(-sign * (field / scale), dtype=np.float32).reshape(self.n, 1)
        return J, h, IsingMetadata(scale, offset, sign)


_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_REAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[dDeE][+-]?[0-9]+)?\Z")


class _Reader:
    def __init__(self, text: str, source: str):
        self.source = source
        self.records = []
        for number, line in enumerate(text.splitlines(), 1):
            # Marked inline comments as well as bare trailing comments are legal.
            line = re.split(r"[!%#]", line, maxsplit=1)[0].strip()
            if line:
                self.records.append((number, line.split()))
        self.position = 0
        self.line = 0

    def fail(self, message: str, unsupported: bool = False):
        error = UnsupportedQplibError if unsupported else QplibFormatError
        raise error(f"{self.source}:{self.line}: {message}")

    def record(self, label: str, count: int = 1) -> list[str]:
        if self.position == len(self.records):
            self.fail(f"unexpected end of file while reading {label}")
        self.line, tokens = self.records[self.position]
        self.position += 1
        if len(tokens) < count:
            self.fail(f"{label}: expected {count} values, got {len(tokens)}")
        # Appendix B permits arbitrary comments beyond the required fields.
        return tokens[:count]

    def integer(self, token: str, label: str, low: int, high: int | None = None) -> int:
        if not _INTEGER.fullmatch(token):
            self.fail(f"{label}: invalid integer {token!r}")
        value = int(token)
        if value < low or (high is not None and value > high):
            self.fail(f"{label}: {value} outside [{low}, {high if high is not None else 'inf'}]")
        return value

    def count(self, label: str, high: int | None = None, low: int = 0) -> int:
        return self.integer(self.record(label)[0], label, low, high)

    def real(self, token: str, label: str, infinity: bool = False) -> float:
        if not _REAL.fullmatch(token):
            self.fail(f"{label}: invalid real {token!r}")
        value = float(token.replace("D", "E").replace("d", "e"))
        # The official infinity marker 1.79769313486232E+308 rounds above DBL_MAX.
        if infinity:
            if value <= 0:
                self.fail(f"{label}: infinity marker must be positive")
        elif not math.isfinite(value):
            self.fail(f"{label}: value must be finite")
        return value

    def vector(self, label: str, n: int) -> np.ndarray:
        default = self.real(self.record(f"{label} default")[0], label)
        result = np.full(n, default, dtype=np.float64)
        count = self.count(f"{label} override count", n)
        seen = set()
        for _ in range(count):
            index, value = self.record(f"{label} override", 2)
            index = self.integer(index, f"{label} index", 1, n) - 1
            if index in seen:
                self.fail(f"{label}: duplicate index {index + 1}")
            seen.add(index)
            result[index] = self.real(value, label)
        return result

    def names(self, label: str, n: int):
        count = self.count(f"{label} count", n)
        seen = set()
        for _ in range(count):
            index, _name = self.record(label, 2)
            index = self.integer(index, f"{label} index", 1, n)
            if index in seen:
                self.fail(f"{label}: duplicate index {index}")
            seen.add(index)


def parse_qplib(text: str, *, source: str = "<string>") -> QplibProblem:
    """Parse the website's QPLIB text, rejecting unsupported or malformed data.

    Accepted types: {L,D,C,Q}B{N,B}. Both constraint codes N and B have no
    general constraints; binary bounds 0/1 are implicit and no bound/type
    records occur. L omits the entire quadratic section. All fields through
    the infinity marker are mandatory. The warm-start/name trailer may be
    omitted as a whole; if present it must be complete, even though it is not
    used by the solver. Unknown trailing records and duplicate indices fail.
    """
    reader = _Reader(text, source)
    name = reader.record("problem name")[0]
    problem_type = reader.record("problem type")[0].upper()
    if (len(problem_type) != 3 or problem_type[0] not in "LDCQ"
            or problem_type[1] != "B" or problem_type[2] not in "NB"):
        reader.fail(f"unsupported problem type {problem_type!r}; expected {{L,D,C,Q}}B{{N,B}}", True)
    sense = reader.record("objective sense")[0].lower()
    if sense not in ("minimize", "maximize"):
        reader.fail(f"unsupported objective sense {sense!r}", True)
    n = reader.count("number of variables", low=1)
    count = 0 if problem_type[0] == "L" else reader.count("quadratic count", n * (n + 1) // 2)
    row = np.empty(count, dtype=np.int64)
    col = np.empty(count, dtype=np.int64)
    value = np.empty(count, dtype=np.float64)
    seen = set()
    for position in range(count):
        i, j, coefficient = reader.record("quadratic term", 3)
        i = reader.integer(i, "quadratic row", 1, n) - 1
        j = reader.integer(j, "quadratic column", 1, n) - 1
        if i < j:
            reader.fail("quadratic entries must be lower triangular (row >= column)")
        if (i, j) in seen:
            reader.fail(f"duplicate quadratic entry ({i + 1}, {j + 1})")
        seen.add((i, j))
        coefficient = reader.real(coefficient, "quadratic coefficient")
        if problem_type[0] == "D" and i != j and coefficient != 0:
            reader.fail("diagonal objective type D contains an off-diagonal term")
        row[position], col[position], value[position] = i, j, coefficient
    linear = reader.vector("linear objective", n)
    constant = reader.real(reader.record("objective constant")[0], "objective constant")
    infinity = reader.real(reader.record("infinity marker")[0], "infinity marker", infinity=True)
    if reader.position < len(reader.records):
        reader.vector("primal starting point", n)
        reader.vector("bound dual starting point", n)
        reader.names("variable names", n)
        reader.names("constraint names", 0)
    if reader.position != len(reader.records):
        reader.line = reader.records[reader.position][0]
        reader.fail("unexpected trailing record")
    for array in (row, col, value, linear):
        array.setflags(write=False)
    return QplibProblem(name, problem_type, sense, n, row, col, value, linear, constant, infinity)


def load_qplib(path: str | Path) -> QplibProblem:
    """Read a UTF-8 .qplib file and validate its complete structure."""
    path = Path(path)
    return parse_qplib(path.read_text(encoding="utf-8"), source=str(path))


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    problem: QplibProblem
    size_bytes: int
    downloaded: bool


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def download_qplib(instance_id: int, directory: str | Path = "data", *, timeout: float = 60.0) -> DownloadResult:
    """Fetch one official instance once; validate before atomic, no-clobber publish.

    Valid existing files are reused without a network request. Invalid existing
    files raise an error and are left intact, so removal is a deliberate human
    decision. Errors propagate; temporary files are removed on failure.
    """
    if isinstance(instance_id, bool) or not isinstance(instance_id, Integral) or instance_id <= 0:
        raise ValueError("instance_id must be a positive integer")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    name = f"QPLIB_{instance_id:04d}"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.qplib"

    def validate(candidate):
        problem = load_qplib(candidate)
        if problem.name.upper() != name:
            raise QplibFormatError(f"{candidate}: expected problem name {name}, got {problem.name!r}")
        return problem

    if path.exists() or path.is_symlink():
        problem = validate(path)
        return DownloadResult(path, problem, path.stat().st_size, False)

    temporary = None
    try:
        with build_opener(_NoRedirect()).open(BASE_URL + path.name, timeout=timeout) as response:
            if response.getcode() != 200:
                raise OSError(f"{BASE_URL + path.name}: HTTP status {response.getcode()}")
            size = 0
            with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{name}.", suffix=".part", delete=False) as out:
                temporary = Path(out.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    size += len(chunk)
                out.flush()
                os.fsync(out.fileno())
            length = response.headers.get("Content-Length")
            if length is not None and size != int(length):
                raise OSError(f"{path.name}: Content-Length {length}, received {size} bytes")
        problem = validate(temporary)
        # Atomic on the same filesystem; unlike replace(), link() cannot clobber
        # a destination created by another process while the download was active.
        os.link(temporary, path)
        return DownloadResult(path, problem, size, True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download and validate official CC-BY 4.0 instances")
    download.add_argument("ids", nargs="*", type=int, help="instance IDs; default: all 23 repository benchmarks")
    download.add_argument("--directory", type=Path, default=Path("data"))
    download.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    downloaded_bytes = downloaded_count = reused = 0
    for instance_id in dict.fromkeys(args.ids or DEFAULT_INSTANCE_IDS):
        try:
            result = download_qplib(instance_id, args.directory, timeout=args.timeout)
        except (OSError, ValueError, HTTPException) as exc:
            print(f"FAILED QPLIB_{instance_id}: {exc}", file=sys.stderr, flush=True)
            return 1
        downloaded_count += int(result.downloaded)
        reused += int(not result.downloaded)
        downloaded_bytes += result.size_bytes if result.downloaded else 0
        problem = result.problem
        status = "downloaded" if result.downloaded else "validated existing"
        print(f"{problem.name}: n={problem.n} {problem.problem_type} {problem.sense} "
              f"{result.size_bytes} bytes ({status})", flush=True)
    print(f"Downloaded {downloaded_count} files / {downloaded_bytes} bytes; "
          f"reused {reused}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
