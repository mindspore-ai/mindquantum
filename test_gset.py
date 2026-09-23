"""Gset maximum-cut benchmark with matched starts for both DSB and BSB."""
import numpy as np
from scipy.sparse import coo_matrix

from benchmarking import arguments, benchmark


def read_gset(path):
    with open(path, encoding="utf-8") as source:
        n, edges = map(int, source.readline().split())
        entries = np.loadtxt(source, ndmin=2)
    if n <= 0 or entries.shape != (edges, 3):
        raise ValueError("invalid Gset dimensions or edge count")
    indices = entries[:, :2].astype(np.int64)
    if not np.array_equal(indices, entries[:, :2]) or np.any(indices < 1) or np.any(indices > n):
        raise ValueError("invalid Gset vertex index")
    if not np.isfinite(entries[:, 2]).all():
        raise ValueError("nonfinite Gset weight")
    row, col = indices.T - 1
    weights = entries[:, 2]
    matrix = coo_matrix((np.r_[weights, weights], (np.r_[row, col], np.r_[col, row])), shape=(n, n)).tocsr()
    return matrix


def main():
    parser = arguments(__doc__)
    parser.add_argument("--graph", default="data/gset/G22")
    args = parser.parse_args()
    weights = read_gset(args.graph)

    def cut(binary):
        spin = 2 * binary - 1
        return (weights.sum() - (spin * (weights @ spin)).sum(axis=0)) / 4

    benchmark(args.graph, -weights.toarray(), np.zeros(weights.shape[0]), cut, "maximize", args)


if __name__ == "__main__":
    main()
