"""
Some demos.
"""

from qaoa_search import QAOAConfig
from schnorr_algorithm import SearchType, SchnorrConfig, run_factorize


# demo 1: Normal mode.
schnorr_config = SchnorrConfig(
    cap_n=1961,
    smooth_b1=5,
    smooth_b2=5,
    pwr_range=(0.5, 6),
    search_type=SearchType.NONE,
    qaoa_config=None
)

run_factorize(schnorr_config, verbose=1, outdir="output/")


# demo 2: Brute-Force search.
schnorr_config = SchnorrConfig(
    cap_n=1961,
    smooth_b1=3,
    smooth_b2=2 * 3**2,
    pwr=1.5,
    search_type=SearchType.BF,
    max_iter=int(1e5),
    qaoa_config=None
)

run_factorize(schnorr_config, verbose=1, outdir="output/")


# demo 3: QAOA search.
qaoa_config = QAOAConfig(n_layer=4, verbose=1)
schnorr_config = SchnorrConfig(
    cap_n=1961,
    smooth_b1=3,
    smooth_b2=2 * 3**2,
    pwr=1.5,
    search_type=SearchType.QAOA,
    qaoa_config=qaoa_config,
    max_iter=500,
)

run_factorize(schnorr_config, verbose=1, outdir="output/", log_step=10)
