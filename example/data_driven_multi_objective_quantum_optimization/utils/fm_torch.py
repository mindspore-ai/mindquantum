"""Shared PyTorch Factorization Machine training and FM-to-QUBO conversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from .experiment_runtime import (
    FM_FACTORIZATION_RANK,
    NUMPY_SEED_MAX,
    require_integer,
)
from .experiment_runtime import (
    fm_seed_plan as _fm_seed_plan,
)

FM_MAX_STEPS = 2000
FM_LBFGS_LR = 1.0
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1


@dataclass(frozen=True)
class FMHyperParams:
    """Optuna 或固定配置选出的 FM 超参数。."""

    init_std: float
    l2_reg_w: float
    l2_reg_v: float


@dataclass(frozen=True)
class FMSplitData:
    """FM 训练、验证、测试划分后的数组集合。."""

    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


class TorchFMRegressor(nn.Module):
    """二阶 Factorization Machine 回归器。.

    `w0 + w*x + interaction(V)` 的形式可以直接展开成二次项，因此训练完成后能被
    `fm_to_qubo` 转换为 QUBO。
    """

    def __init__(self, num_features: int, init_std: float) -> None:
        """Initialize the instance."""
        super().__init__()
        self.num_features = require_integer("num_features", num_features, minimum=1)
        self.rank = FM_FACTORIZATION_RANK
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(self.num_features))
        self.V = nn.Parameter(torch.empty(self.num_features, self.rank))
        nn.init.normal_(self.V, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_term = self.w0 + torch.matmul(x, self.w)
        vx = torch.matmul(x, self.V)
        x_square = x * x
        v_square = self.V * self.V
        interaction_term = 0.5 * torch.sum((vx * vx) - torch.matmul(x_square, v_square), dim=1)
        return linear_term + interaction_term


def set_global_seed(seed: int) -> None:
    """Run set global seed."""
    seed = require_integer("seed", seed, minimum=0, maximum=NUMPY_SEED_MAX)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_validation_test(x: np.ndarray, y: np.ndarray, seed: int) -> FMSplitData:
    """按论文复现流程拆分当前 active-learning 数据集。."""
    seed = require_integer("seed", seed, minimum=0, maximum=NUMPY_SEED_MAX - 1)
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if len(x) < 5:
        return FMSplitData(x, y, x, y, x, y)

    num_samples = len(x)
    train_size = max(1, min(int(num_samples * TRAIN_RATIO), num_samples - 2))
    temporary_size = num_samples - train_size
    validation_fraction = VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO)
    validation_size = max(1, min(int(temporary_size * validation_fraction), temporary_size - 1))
    test_size = temporary_size - validation_size

    train_x, temp_x, train_y, temp_y = train_test_split(
        x,
        y,
        train_size=train_size,
        test_size=temporary_size,
        random_state=seed,
        shuffle=True,
    )
    validation_x, test_x, validation_y, test_y = train_test_split(
        temp_x,
        temp_y,
        train_size=validation_size,
        test_size=test_size,
        random_state=seed + 1,
        shuffle=True,
    )
    return FMSplitData(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        test_x=test_x,
        test_y=test_y,
    )


def _tensor_pair(x: np.ndarray, y: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(x.astype(np.float32)).to(device), torch.from_numpy(y.astype(np.float32)).to(device)


def _evaluate_loss(
    model: TorchFMRegressor,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
) -> float:
    with torch.no_grad():
        prediction = model(x_tensor)
        return float(torch.mean((prediction - y_tensor) ** 2).item())


def _fit_model_once(
    split_data: FMSplitData,
    hparams: FMHyperParams,
    device: str,
    seed: int,
    *,
    include_test_loss: bool,
) -> tuple[TorchFMRegressor, dict[str, float]]:
    """用一组超参数训练一次 FM，并返回本次调用所需的数据集 loss。."""
    set_global_seed(seed)
    model = TorchFMRegressor(num_features=split_data.train_x.shape[1], init_std=hparams.init_std).to(device)
    train_x_tensor, train_y_tensor = _tensor_pair(split_data.train_x, split_data.train_y, device)
    validation_x_tensor, validation_y_tensor = _tensor_pair(split_data.validation_x, split_data.validation_y, device)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=FM_LBFGS_LR,
        max_iter=1,
        line_search_fn="strong_wolfe",
    )

    best_train_loss = float("inf")
    stagnation_steps = 0
    for _ in range(FM_MAX_STEPS):

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            prediction = model(train_x_tensor)
            mse = torch.mean((prediction - train_y_tensor) ** 2)
            reg_w = hparams.l2_reg_w * torch.sum(model.w * model.w)
            reg_v = hparams.l2_reg_v * torch.sum(model.V * model.V)
            total_loss = mse + reg_w + reg_v
            total_loss.backward()
            return total_loss

        loss = float(optimizer.step(closure).item())
        if loss + 1e-10 < best_train_loss:
            best_train_loss = loss
            stagnation_steps = 0
        else:
            stagnation_steps += 1
        if stagnation_steps >= 25:
            break

    metrics = {
        "train_loss": best_train_loss,
        "validation_loss": _evaluate_loss(model, validation_x_tensor, validation_y_tensor),
    }
    if include_test_loss:
        test_x_tensor, test_y_tensor = _tensor_pair(split_data.test_x, split_data.test_y, device)
        metrics["test_loss"] = _evaluate_loss(model, test_x_tensor, test_y_tensor)
    return model, metrics


def tune_fm_hparams(
    split_data: FMSplitData,
    optuna_trials: int,
    device: str,
    seed: int,
) -> FMHyperParams:
    """只选择并返回 FM 超参数；最终模型由调用方训练一次。."""
    optuna_trials = require_integer("optuna_trials", optuna_trials, minimum=0)
    seed_plan = _fm_seed_plan(seed, optuna_trials)
    default_hparams = FMHyperParams(
        init_std=0.05,
        l2_reg_w=1e-4,
        l2_reg_v=1e-4,
    )
    if optuna_trials <= 0:
        return default_hparams

    sampler = optuna.samplers.TPESampler(seed=seed_plan["tuner"])
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        hparams = FMHyperParams(
            init_std=trial.suggest_float("init_std", 1e-3, 5e-1, log=True),
            l2_reg_w=trial.suggest_float("l2_reg_w", 1e-8, 1e-1, log=True),
            l2_reg_v=trial.suggest_float("l2_reg_v", 1e-8, 1e-1, log=True),
        )
        _, metrics = _fit_model_once(
            split_data,
            hparams,
            device,
            seed_plan["trials"][trial.number],
            include_test_loss=False,
        )
        return metrics["validation_loss"]

    study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
    best = study.best_trial.params
    return FMHyperParams(
        init_std=float(best["init_std"]),
        l2_reg_w=float(best["l2_reg_w"]),
        l2_reg_v=float(best["l2_reg_v"]),
    )


def fit_torch_fm(
    x: np.ndarray,
    y: np.ndarray,
    optuna_trials: int,
    device: str,
    seed: int,
) -> tuple[TorchFMRegressor, dict[str, Any]]:
    """训练当前迭代的 FM，并返回可写入 checkpoint/summary 的训练 metadata。."""
    optuna_trials = require_integer("optuna_trials", optuna_trials, minimum=0)
    seed_plan = _fm_seed_plan(seed, optuna_trials)
    split_data = split_train_validation_test(x, y, seed_plan["root"])
    best_hparams = tune_fm_hparams(
        split_data,
        optuna_trials=optuna_trials,
        device=device,
        seed=seed_plan["root"],
    )
    model, final_metrics = _fit_model_once(
        split_data,
        best_hparams,
        device,
        seed_plan["final_fit"],
        include_test_loss=True,
    )
    metadata: dict[str, Any] = {
        "tuner": "optuna" if optuna_trials > 0 else "fixed",
        "rank": FM_FACTORIZATION_RANK,
        "seed_plan": seed_plan,
        **asdict(best_hparams),
        **final_metrics,
    }
    return model, metadata


def fm_to_qubo(model: TorchFMRegressor) -> tuple[np.ndarray, float]:
    """把训练好的 FM 展开为 QUBO，并按论文补充材料 S1.1 丢弃整体偏置。."""
    w = model.w.detach().cpu().numpy().astype(np.float64)
    v = model.V.detach().cpu().numpy().astype(np.float64)
    q = 0.5 * (v @ v.T)
    np.fill_diagonal(q, w)
    return q, 0.0


__all__ = [
    "FMHyperParams",
    "FMSplitData",
    "NUMPY_SEED_MAX",
    "TorchFMRegressor",
    "fit_torch_fm",
    "fm_to_qubo",
    "split_train_validation_test",
    "tune_fm_hparams",
]
