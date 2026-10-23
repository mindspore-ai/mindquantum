"""CPU-only tests: no NPU, MindQuantum, SciPy, or network access required."""

from http.client import IncompleteRead
from io import BytesIO
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

import qplib


# T is stored only once, not reflected: e.g. the -10 entry means -5*z2*z1.
# Variable 5 has no quadratic entries, and an explicit zero is retained.
CORE = """QPLIB_3506
QBN
Minimize
5
6
1 1 6.0
2 1 -10.0
2 2 -4.0
3 1 3.5
4 2 8.0
4 3 0.0
1.25
2
2 -2.5
3 0.0
7.5
1.79769313486232E+308
"""
TRAILER = """0.0
1
5 1.0
0.0
1
2 -0.75
2
1 first
5 isolated
0
"""
SMALL = CORE + TRAILER


def ising_energies(J, h, spins):
    # Compute energy in float64 to separate coefficient-rounding from solver math.
    J, h = J.astype(np.float64), h.astype(np.float64)
    return -0.5 * np.sum(spins * (J @ spins), axis=0) - (h.T @ spins).ravel()


@pytest.mark.parametrize("sense,sign", [("Minimize", 1), ("Maximize", -1)])
def test_exhaustive_original_objective_and_ising(sense, sign, monkeypatch):
    problem = qplib.parse_qplib(SMALL.replace("Minimize", sense))
    binary = np.array(list(product((0, 1), repeat=5)), dtype=np.float64).T
    # Independent hand-written polynomial: never use parsed coefficients, a
    # mirrored matrix, the conversion, or metadata to construct this oracle.
    expected = np.array([
        3*a*a - 5*a*b - 2*b*b + 1.75*a*c + 4*b*d
        + 1.25*a - 2.5*b + 1.25*d + 1.25*e + 7.5
        for a, b, c, d, e in binary.T
    ])
    np.testing.assert_array_equal(problem.objective(binary), expected)
    np.testing.assert_array_equal(problem.linear, [1.25, -2.5, 0, 1.25, 1.25])
    assert problem.n == 5 and problem.q_row.max() == 3
    assert problem.q_value[-1] == 0
    assert not problem.q_value.flags.writeable
    assert not problem.linear.flags.writeable

    J, h, metadata = problem.to_ising()
    assert J.dtype == h.dtype == np.float32
    assert J.shape == (5, 5) and h.shape == (5, 1)
    assert J.flags.c_contiguous and h.flags.c_contiguous
    np.testing.assert_array_equal(J, J.T)
    np.testing.assert_array_equal(np.diag(J), 0)
    np.testing.assert_array_equal(J[4], 0)
    assert h[4, 0] != 0  # isolated variable still receives its linear default
    assert max(np.abs(J).max(), np.abs(h).max()) == 1
    assert metadata.sense_sign == sign
    assert metadata.scale == 2.5
    assert metadata.offset == 8.8125
    energies = ising_energies(J, h, 2*binary - 1)
    np.testing.assert_allclose(energies, sign*(expected - 8.8125)/2.5, rtol=0, atol=3e-7)
    np.testing.assert_allclose(metadata.to_objective(energies), expected, rtol=0, atol=1e-6)
    # Check all pairwise orderings, ties, and the optimizer, not merely shapes.
    objective_difference = sign*(expected[:, None] - expected[None, :])
    energy_difference = energies[:, None] - energies[None, :]
    unequal = objective_difference != 0
    np.testing.assert_array_equal(np.sign(energy_difference[unequal]), np.sign(objective_difference[unequal]))
    np.testing.assert_allclose(energy_difference[~unequal], 0, atol=6e-7)
    assert expected[np.argmin(energies)] == (expected.min() if sign == 1 else expected.max())
    # Objective evaluation must not accidentally depend on the solver conversion.
    monkeypatch.setattr(qplib.QplibProblem, "to_ising", Mock(side_effect=AssertionError("not independent")))
    np.testing.assert_array_equal(problem.objective(binary.astype(bool)), expected)


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
@pytest.mark.parametrize("default", [0.0, 3.25])
@pytest.mark.parametrize("kind", ["QBN", "LBN", "DBN", "CBN", "QBB"])
def test_no_quadratic_entries_no_linear_overrides(sense, default, kind):
    quadratic_count = "" if kind[0] == "L" else "0\n"
    text = f"empty\n{kind}\n{sense}\n3\n{quadratic_count}{default}\n0\n-7.25\n1e20\n"
    problem = qplib.parse_qplib(text)
    binary = np.array(list(product((0, 1), repeat=3)), dtype=float).T
    expected = default*binary.sum(axis=0) - 7.25
    np.testing.assert_array_equal(problem.objective(binary), expected)
    J, h, metadata = problem.to_ising()
    np.testing.assert_array_equal(J, 0)
    assert metadata.scale == (default/2 or 1)
    np.testing.assert_allclose(metadata.to_objective(ising_energies(J, h, 2*binary - 1)), expected)
    assert problem.objective(np.empty((3, 0))).shape == (0,)


def test_diagonal_terms_become_fields_not_self_couplings():
    problem = qplib.parse_qplib("diagonal\nDBN\nminimize\n2\n2\n1 1 4\n2 2 -8\n0\n0\n3\n1e20\n")
    J, h, metadata = problem.to_ising()
    np.testing.assert_array_equal(J, 0)
    np.testing.assert_array_equal(h.ravel(), [-0.5, 1.0])
    assert metadata.scale == 2 and metadata.offset == 2
    binary = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    np.testing.assert_array_equal(problem.objective(binary), [3, -1, 5, 1])
    np.testing.assert_array_equal(metadata.to_objective(ising_energies(J, h, 2*binary - 1)), [3, -1, 5, 1])


def test_comments_exponents_case_and_record_order(tmp_path):
    text = SMALL.replace("QBN", "qbn").replace("Minimize", "mInImIzE")
    text = text.replace("1.25\n", "1.25d+0 linear default\n").replace("2 1 -10.0", "2 1 -1D1")
    text = "\n# heading\n   ! comment\n% comment\n" + "\n\n".join(
        line + " # arbitrary explanation" for line in text.splitlines()
    )
    path = tmp_path / "problem.qplib"
    path.write_text(text, encoding="utf-8")
    problem = qplib.load_qplib(path)
    assert problem.sense == "minimize" and problem.problem_type == "QBN"
    assert problem.infinity == float("inf")  # official marker rounds above DBL_MAX
    np.testing.assert_array_equal(problem.q_value, [6, -10, -4, 3.5, 8, 0])
    # Unsorted records are valid; sorting is not required to interpret defaults.
    reordered = SMALL.replace("1 1 6.0\n2 1 -10.0", "2 1 -10.0\n1 1 6.0")
    np.testing.assert_array_equal(qplib.parse_qplib(reordered).objective(np.ones((5, 1))), problem.objective(np.ones((5, 1))))


@pytest.mark.parametrize("kind", ["QBL", "QBQ", "QBD", "QBC", "QCN", "QIN", "QMN", "QGN", "XBN", "QB", "QBNX"])
def test_unsupported_types_are_explicit(kind):
    with pytest.raises(qplib.UnsupportedQplibError, match="unsupported problem type"):
        qplib.parse_qplib(SMALL.replace("QBN", kind))


@pytest.mark.parametrize("sense", ["min", "max", "satisfy", "0"])
def test_unsupported_senses_are_explicit(sense):
    with pytest.raises(qplib.UnsupportedQplibError, match="unsupported objective sense"):
        qplib.parse_qplib(SMALL.replace("Minimize", sense))


@pytest.mark.parametrize("old,new,match", [
    ("\n5\n6\n", "\n0\n6\n", "number of variables"),
    ("\n5\n6\n", "\n-1\n6\n", "number of variables"),
    ("\n5\n6\n", "\n5.0\n6\n", "invalid integer"),
    ("\n6\n1 1", "\n-1\n1 1", "quadratic count"),
    ("\n6\n1 1", "\n16\n1 1", "quadratic count"),
    ("1 1 6.0", "0 1 6.0", "quadratic row"),
    ("1 1 6.0", "6 1 6.0", "quadratic row"),
    ("1 1 6.0", "1 0 6.0", "quadratic column"),
    ("1 1 6.0", "1 6 6.0", "quadratic column"),
    ("2 1 -10.0", "1 2 -10.0", "lower triangular"),
    ("2 1 -10.0", "1 1 -10.0", "duplicate quadratic"),
    ("2 1 -10.0", "2 1", "expected 3 values"),
    ("2 1 -10.0", "2 1 __import__('os')", "invalid real"),
    ("1.25\n2\n", "1.25\n6\n", "linear objective override count"),
    ("2 -2.5", "0 -2.5", "linear objective index"),
    ("2 -2.5", "6 -2.5", "linear objective index"),
    ("\n3 0.0\n", "\n2 0.0\n", "linear objective: duplicate"),
    ("1.79769313486232E+308", "0", "infinity marker"),
    ("1.79769313486232E+308", "-1e20", "infinity marker"),
    ("5 1.0", "6 1.0", "primal starting point index"),
    ("2 -0.75", "0 -0.75", "bound dual starting point index"),
    ("2 -0.75", "2 nan", "invalid real"),
    ("5 isolated", "6 isolated", "variable names index"),
    ("5 isolated", "1 isolated", "duplicate index"),
    ("5 isolated", "5 # missing name", "expected 2 values"),
    ("5 isolated\n0", "5 isolated\n1\n1 constraint", "constraint names count"),
    ("QBN", "DBN", "off-diagonal term"),
])
def test_malformed_records(old, new, match):
    assert old in SMALL
    with pytest.raises(qplib.QplibFormatError, match=match):
        qplib.parse_qplib(SMALL.replace(old, new), source="broken.qplib")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e400", "__import__('os')"])
@pytest.mark.parametrize("old", ["1 1 6.0", "1.25", "2 -2.5", "7.5"])
def test_nonfinite_or_executable_objective_tokens_fail(value, old):
    replacement = " ".join(old.split()[:-1] + [value])
    with pytest.raises(qplib.QplibFormatError):
        qplib.parse_qplib(SMALL.replace(old, replacement))


def test_all_truncations_and_trailing_garbage_fail():
    lines = SMALL.splitlines()
    for count in range(len(lines)):
        # The only optional omission is the entire warm-start/name trailer.
        if count == len(CORE.splitlines()):
            qplib.parse_qplib("\n".join(lines[:count]))
        else:
            with pytest.raises(qplib.QplibFormatError, match="end of file"):
                qplib.parse_qplib("\n".join(lines[:count]))
    with pytest.raises(qplib.QplibFormatError, match="unexpected trailing record"):
        qplib.parse_qplib(SMALL + "0\n")


@pytest.mark.parametrize("binary", [
    np.zeros(5), np.zeros((1, 5)), np.zeros((5, 1, 1)),
    np.full((5, 1), 0.5), np.full((5, 1), -1), np.full((5, 1), np.nan),
    np.full((5, 1), np.inf), np.full((5, 1), "0"), np.zeros((5, 1), dtype=complex),
])
def test_invalid_objective_input(binary):
    with pytest.raises(ValueError, match="binary"):
        qplib.parse_qplib(SMALL).objective(binary)


def test_conversion_overflow_is_explicit():
    problem = qplib.parse_qplib(CORE.replace("1.25\n", "1.7e308\n").replace("7.5\n", "1.7e308\n"))
    with pytest.raises(ValueError, match="overflows"):
        problem.to_ising()


class Response(BytesIO):
    def __init__(self, payload, *, length=None, status=200):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload) if length is None else length)}
        self.status = status

    def getcode(self):
        return self.status


def fake_network(monkeypatch, *, response=None, error=None):
    request = Mock(return_value=response, side_effect=error)
    factory = Mock(return_value=SimpleNamespace(open=request))
    monkeypatch.setattr(qplib, "build_opener", factory)
    return request, factory


def test_download_validates_before_atomic_publish_and_reuses(tmp_path, monkeypatch):
    payload = SMALL.encode()
    request, factory = fake_network(monkeypatch, response=Response(payload))
    real_link = qplib.os.link
    published = []

    def checked_link(source, destination):
        assert not destination.exists()
        assert source.parent == destination.parent
        assert source.read_bytes() == payload
        assert qplib.load_qplib(source).n == 5
        published.append(destination)
        real_link(source, destination)

    monkeypatch.setattr(qplib.os, "link", checked_link)
    result = qplib.download_qplib(3506, tmp_path, timeout=12)
    assert result.downloaded and result.size_bytes == len(payload)
    assert result.path.read_bytes() == payload and result.problem.n == 5
    assert list(tmp_path.iterdir()) == published
    request.assert_called_once_with("https://qplib.zib.de/qplib/QPLIB_3506.qplib", timeout=12)
    handler = factory.call_args.args[0]
    assert handler.redirect_request(None, None, 302, None, None, "https://mirror.invalid") is None
    before = result.path.stat()
    again = qplib.download_qplib(3506, tmp_path)
    assert not again.downloaded
    assert again.path.stat().st_mtime_ns == before.st_mtime_ns
    assert again.path.stat().st_ino == before.st_ino
    request.assert_called_once()  # no second network access, not even HEAD


@pytest.mark.parametrize("payload", [
    b"<html>not data</html>", CORE.rsplit("\n", 2)[0].encode(),
    SMALL.replace("QPLIB_3506", "QPLIB_3565").encode(),
    SMALL.replace("QBN", "QBL").encode(), (SMALL + "garbage\n").encode(),
])
def test_invalid_download_is_not_published(tmp_path, monkeypatch, payload):
    request, _ = fake_network(monkeypatch, response=Response(payload))
    with pytest.raises(qplib.QplibFormatError):
        qplib.download_qplib(3506, tmp_path)
    request.assert_called_once()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("payload", [b"broken", SMALL.replace("QPLIB_3506", "other").encode()])
def test_invalid_existing_file_is_never_overwritten(tmp_path, monkeypatch, payload):
    path = tmp_path / "QPLIB_3506.qplib"
    path.write_bytes(payload)
    request, _ = fake_network(monkeypatch, error=AssertionError("network must not be used"))
    with pytest.raises(qplib.QplibFormatError):
        qplib.download_qplib(3506, tmp_path)
    assert path.read_bytes() == payload and list(tmp_path.iterdir()) == [path]
    request.assert_not_called()


@pytest.mark.parametrize("error", [
    URLError("offline"), TimeoutError("timeout"),
    HTTPError(qplib.BASE_URL, 503, "service unavailable", {}, None),
    HTTPError(qplib.BASE_URL, 302, "redirect refused", {}, None),
])
def test_download_failure_has_no_retry_or_mirror(tmp_path, monkeypatch, error):
    request, _ = fake_network(monkeypatch, error=error)
    with pytest.raises(type(error)):
        qplib.download_qplib(3506, tmp_path)
    request.assert_called_once()
    assert list(tmp_path.iterdir()) == []


def test_incomplete_transfer_cleans_temporary_file(tmp_path, monkeypatch):
    response = Response(SMALL.encode())
    response.read = Mock(side_effect=[b"partial", IncompleteRead(b"", 100)])
    request, _ = fake_network(monkeypatch, response=response)
    with pytest.raises(IncompleteRead):
        qplib.download_qplib(3506, tmp_path)
    request.assert_called_once()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kwargs,match", [({"length": 100000}, "Content-Length"), ({"status": 206}, "HTTP status")])
def test_bad_http_response(tmp_path, monkeypatch, kwargs, match):
    request, _ = fake_network(monkeypatch, response=Response(SMALL.encode(), **kwargs))
    with pytest.raises(OSError, match=match):
        qplib.download_qplib(3506, tmp_path)
    request.assert_called_once()
    assert list(tmp_path.iterdir()) == []


def test_publish_race_does_not_clobber_existing_file(tmp_path, monkeypatch):
    fake_network(monkeypatch, response=Response(SMALL.encode()))
    real_link = qplib.os.link

    def racing_link(source, destination):
        destination.write_bytes(b"other process owns this file")
        real_link(source, destination)

    monkeypatch.setattr(qplib.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        qplib.download_qplib(3506, tmp_path)
    path = tmp_path / "QPLIB_3506.qplib"
    assert path.read_bytes() == b"other process owns this file"
    assert list(tmp_path.iterdir()) == [path]


def test_cli_reports_first_failure_and_stops(tmp_path, monkeypatch, capsys):
    request, _ = fake_network(monkeypatch)
    request.side_effect = [URLError("offline"), Response(SMALL.replace("3506", "3565").encode())]
    assert qplib.main(["download", "3506", "3565", "3506", "--directory", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "FAILED QPLIB_3506" in captured.err and "offline" in captured.err
    assert captured.out == ""
    request.assert_called_once()


def test_cli_defaults_to_exact_benchmark_set(tmp_path, monkeypatch, capsys):
    expected = (3506, 3565, 3642, 3650, 3693, 3705, 3706, 3738, 3745, 3822, 3832, 3838,
                3850, 3852, 3877, 5721, 5725, 5755, 5875, 5881, 5882, 5909, 5922)
    download = Mock(return_value=qplib.DownloadResult(tmp_path, qplib.parse_qplib(SMALL), 10, True))
    monkeypatch.setattr(qplib, "download_qplib", download)
    assert qplib.main(["download", "--directory", str(tmp_path)]) == 0
    assert tuple(call.args[0] for call in download.call_args_list) == expected
    assert "Downloaded 23 files / 230 bytes" in capsys.readouterr().out


@pytest.mark.parametrize("instance_id", [0, -1, True, "3506", "../3506", 3506.5])
def test_invalid_download_ids(tmp_path, monkeypatch, instance_id):
    request, _ = fake_network(monkeypatch)
    with pytest.raises(ValueError, match="instance_id"):
        qplib.download_qplib(instance_id, tmp_path)
    request.assert_not_called()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout(tmp_path, monkeypatch, timeout):
    request, _ = fake_network(monkeypatch)
    with pytest.raises(ValueError, match="timeout"):
        qplib.download_qplib(3506, tmp_path, timeout=timeout)
    request.assert_not_called()


@pytest.mark.parametrize("instance_id", qplib.DEFAULT_INSTANCE_IDS)
def test_local_benchmark_affine_identity(instance_id):
    path = Path(__file__).resolve().parents[1] / "data" / f"QPLIB_{instance_id}.qplib"
    if not path.exists():
        pytest.skip("optional ignored benchmark data not downloaded")
    problem = qplib.load_qplib(path)
    assert problem.name == f"QPLIB_{instance_id}"
    assert problem.problem_type == "QBN"
    binary = np.random.default_rng(instance_id).integers(0, 2, (problem.n, 5))
    binary[:, 0], binary[:, 1] = 0, 1
    J, h, metadata = problem.to_ising()
    np.testing.assert_array_equal(J, J.T)
    np.testing.assert_array_equal(np.diag(J), 0)
    # Use a coefficient-wise float32 rounding bound, not a relative tolerance
    # on f: f can be zero after cancellation of million-unit contributions.
    # For |s_i|=1, |delta E| <= sum(i>j, |delta J_ij|) + sum(|delta h_i|).
    roundoff = np.finfo(np.float32).eps * metadata.scale * (
        0.5 * np.abs(J).sum(dtype=np.float64) + np.abs(h).sum(dtype=np.float64)
    )
    np.testing.assert_allclose(
        metadata.to_objective(ising_energies(J, h, 2*binary - 1)),
        problem.objective(binary), rtol=0, atol=roundoff + 1e-9,
    )
    if instance_id == 3506:
        # Independent authoritative AMPL sample: +2*b2*b3, with b2/b3
        # corresponding to file variables 1/2. This catches factor-of-two bugs.
        z = np.zeros((problem.n, 1))
        z[:2] = 1
        np.testing.assert_array_equal(problem.objective(z), [2.0])
