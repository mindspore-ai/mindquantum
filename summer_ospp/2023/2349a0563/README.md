# SQIF

An implementation of paper: [Factoring integers with sublinear resources on a superconducting quantum processor](https://arxiv.org/abs/2212.12372).

## Install & Run

A Linux system is recommended.

```python
conda env create -f dependencies.yaml -n sqif
conda activate sqif
python sqif/main.py
```

## Example

Here giving 3 demos of factorizing integer. You can modify it to factorize other integer (`cap_n`).

```python
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
```

## Configuration

```python
@dataclass
class QAOAConfig:
    """Configuration for `QAOASearch`"""
    n_layer: int = 4                # Layers of QAOA circuit.
    # Optimizer for QAOA, only support Adam now.
    optimizer: str = "adam"
    learning_rate: float = 0.002    # Learning rate of optimizer.
    max_iter: int = 1000            # Number of iterations in QAOA.
    n_sample_shot: int = 1000       # Number of samples.
    # Output the QAOA debug training information if `verbose` greater than 0.
    verbose: int = 0


@unique
class SearchType(Enum):
    """The search type to optimize Schnorr's algorithm."""
    QAOA = "QAOA"
    BF = "BF"
    NONE = "NONE"


@dataclass
class SchnorrConfig:
    cap_n: int               # The integer N that will be factorized.
    smooth_b1: int = 5       # The number of primes, e.g it's 4, means the maximum prime is 7
    # since the prime list is [2, 3, 5, 7, 11, ...].
    # The second number of primes, in paper it's `2 * (smooth_b1^2)`, make sure `smooth_b2 >= smooth_b1`.
    smooth_b2: int = 5
    max_iter: int = 1e6      # The maximum number of iterations
    # The number of smooth-pair that will sample, if None, it will be set as 2*smooth_b2
    n_pair: int = None
    base: int = 10           # The base used in the last line of CVP matrix.
    pwr: float = None        # The power used in the last line of CVP matrix.
    # Random sample `pwr` from `pwr_range` when `pwr` is None.
    pwr_range: Tuple = (0.5, 6.0)
    # Search type that optimizes the root of babai's algorithm.
    search_type: SearchType = SearchType.NONE
    # Only valid when `search_type` is `SearchType.QAOA`.
    qaoa_config: QAOAConfig = None
```

More details can be seen in project-report.

## Reference

[1] [Bao Yan, Ziqi Tan, Shijie Wei, Haocong Jiang, Weilong Wang, Hong Wang, Lan Luo, Qianheng Duan, Yiting Liu, Wenhao Shi, Yangyang Fei, Xiangdong Meng, Yu Han, Zheng Shan, Jiachen Chen, Xuhao Zhu, Chuanyu Zhang, Feitong Jin, Hekang Li, Chao Song, Zhen Wang, Zhi Ma, H. Wang, and Gui-Lu Long. Factoring integers with sublinear resources on a superconducting quantum processor.](https://arxiv.org/abs/2212.12372)
