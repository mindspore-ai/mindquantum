"""QPLIB benchmark. Fetch data first with: python qplib.py download."""
from pathlib import Path

from benchmarking import arguments, benchmark
from qplib import DEFAULT_INSTANCE_IDS, load_qplib


def main():
    parser = arguments(__doc__)
    parser.add_argument("--instances", nargs="+", type=int, default=DEFAULT_INSTANCE_IDS)
    parser.add_argument("--directory", type=Path, default=Path("data"))
    args = parser.parse_args()
    for instance in args.instances:
        problem = load_qplib(args.directory / f"QPLIB_{instance}.qplib")
        J, h, _ = problem.to_ising()
        benchmark(problem.name, J, h, problem.objective, problem.sense, args)


if __name__ == "__main__":
    main()
