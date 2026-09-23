"""Synthetic dense Ising benchmark. Correctness tests live under tests/."""
import numpy as np

from benchmarking import arguments, benchmark


def main():
    parser = arguments(__doc__)
    parser.set_defaults(cpu="dense")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("size must be positive")
    rng = np.random.default_rng(args.seed)
    upper = np.triu(rng.choice([-1., 1.], size=(args.size, args.size)), 1)
    J = (upper + upper.T).astype(np.float32)

    def energy(binary):
        spin = 2 * binary - 1
        return -.5 * (spin * (J @ spin)).sum(axis=0)

    benchmark(f"dense{args.size}", J, np.zeros(args.size), energy, "minimize", args)


if __name__ == "__main__":
    main()
