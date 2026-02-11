"""
使用Optuna自动调优SB算法的超参数
支持两种算法：BSB、DSB
目标：找到最优的xi、dt、batch_size等参数以接近真实最优解
"""

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import scipy.sparse as sp
import torch
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import json
from datetime import datetime
from pathlib import Path
from qplib_parser import QPLIBParser, ProblemModel


class SBOptimizer:
    def __init__(self, file_path, device="cuda", target_value=None, algorithm="BSB"):
        """
        SB算法优化器（支持BSB、DSB）

        参数:
            file_path: QPLIB文件路径
            device: 'cuda' 或 'cpu'
            target_value: 已知的最优值(用于计算gap)
            algorithm: 算法类型，'BSB' 或 'DSB'
        """
        self.algorithm = algorithm.upper()
        if self.algorithm not in ["BSB", "DSB"]:
            raise ValueError(f"不支持的算法类型: {algorithm}，请选择 'BSB' 或 'DSB'")

        print("=" * 70)
        print(f"🔧 初始化{self.algorithm}优化器")
        print("=" * 70)

        # 解析QPLIB文件
        print(f"\n📂 加载文件: {file_path}")
        model = QPLIBParser.parse_qplib_file(file_path)

        self.problem_name = model.problem_name
        self.objective_sense = model.objective_sense
        self.num_vars = model.num_vars
        self.num_constraints = model.num_constraints
        self.constant_term = model.constant_term
        self.target_value = target_value
        self.device = device

        # 构建二次项稀疏矩阵
        if model.quad_terms_indices and model.quad_terms_coeffs:
            row_indices = [idx[0] for idx in model.quad_terms_indices]
            col_indices = [idx[1] for idx in model.quad_terms_indices]
            values = model.quad_terms_coeffs

            quad_terms = np.array(
                [[row_indices[i], col_indices[i], values[i]] for i in range(len(values))], dtype=np.float32
            )

            diag_index = np.nonzero(quad_terms[:, 0] == quad_terms[:, 1])[0]
            diag_terms = None
            if len(diag_index) > 0:
                diag_terms = quad_terms[diag_index]
                quad_terms = np.delete(quad_terms, diag_index, axis=0)

            quad_terms_sparse = sp.csr_matrix(
                (
                    np.concatenate([quad_terms[:, 2] / 2.0, quad_terms[:, 2] / 2.0]),
                    (
                        np.concatenate([quad_terms[:, 0], quad_terms[:, 1]]),
                        np.concatenate([quad_terms[:, 1], quad_terms[:, 0]]),
                    ),
                ),
                shape=(self.num_vars, self.num_vars),
            )
        else:
            quad_terms_sparse = sp.csr_matrix((self.num_vars, self.num_vars))
            diag_terms = None

        # 转换为PyTorch稀疏张量
        crow_indices = torch.tensor(quad_terms_sparse.indptr, dtype=torch.int32)
        col_indices = torch.tensor(quad_terms_sparse.indices, dtype=torch.int32)
        values = torch.tensor(quad_terms_sparse.data, dtype=torch.float)
        self.quad_terms = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, size=quad_terms_sparse.shape, dtype=torch.float, device=device
        )

        # 处理线性项
        if model.linear_terms is not None:
            self.linear_terms = torch.from_numpy(model.linear_terms).float().to(device)
            if diag_terms is not None:
                self.linear_terms[diag_terms[:, 0]] = self.linear_terms[diag_terms[:, 0]] + diag_terms[:, 2] / 2
        else:
            if diag_terms is None:
                self.linear_terms = None
            else:
                self.linear_terms = torch.zeros(self.num_vars, device=device, dtype=torch.float32)
                self.linear_terms[diag_terms[:, 0]] = diag_terms[:, 2] / 2

        # 计算缩放因子
        if self.linear_terms is not None:
            h_max = torch.abs(self.linear_terms).max().cpu().item()
        else:
            h_max = 0
        self.scaling = np.max([np.abs(quad_terms_sparse.data).max() if len(quad_terms_sparse.data) > 0 else 0, h_max])
        if self.scaling == 0:
            self.scaling = 1.0

        # 是否为纯线性问题
        self.is_linear = len(quad_terms_sparse.data) == 0

        if self.is_linear:
            print("✓ 检测到纯线性问题")
        else:
            # 缩放二次项和线性项
            self.quad_terms.values().div_(self.scaling)
            if self.linear_terms is not None:
                self.linear_terms /= self.scaling

        # 计算默认xi值(作为参考)
        if self.is_linear:
            self.default_xi = 0.5 * np.sqrt(self.num_vars - 1) / torch.sqrt((self.linear_terms**2).sum()).item()
        else:
            if self.linear_terms is not None:
                self.default_xi = (
                    0.5
                    * np.sqrt(self.num_vars - 1)
                    / torch.sqrt((self.quad_terms.values() ** 2).sum() + (self.linear_terms**2).sum()).item()
                )
            else:
                self.default_xi = (
                    0.5 * np.sqrt(self.num_vars - 1) / torch.sqrt((self.quad_terms.values() ** 2).sum()).item()
                )

        print(f"\n📊 问题统计:")
        print(f"  - 问题名称: {self.problem_name}")
        print(f"  - 优化方向: {self.objective_sense}")
        print(f"  - 变量数量: {self.num_vars}")
        print(f"  - 二次项数量: {len(quad_terms_sparse.data)}")
        print(f"  - 线性项数量: {len(self.linear_terms) if self.linear_terms is not None else 0}")
        print(f"  - 缩放因子: {self.scaling:.6f}")
        print(f"  - 默认xi值: {self.default_xi:.6f}")
        if self.target_value is not None:
            print(f"  - 目标最优值: {self.target_value}")

    def calc_grad(self, x):
        """计算目标函数的梯度"""
        if x.dim() == 1:
            x_matrix = x.unsqueeze(1)
            quad_grad = torch.sparse.mm(self.quad_terms, x_matrix).squeeze(1)
        else:
            quad_grad = torch.sparse.mm(self.quad_terms, x)

        if self.linear_terms is not None:
            if x.dim() == 2:
                grad = quad_grad + self.linear_terms.unsqueeze(1).expand_as(quad_grad)
            else:
                grad = quad_grad + self.linear_terms
        else:
            grad = quad_grad

        return grad

    def calc_energy(self, x):
        """计算目标函数值"""
        x_float = x.float()
        scaled_energy = 0.5 * (self.quad_terms @ x_float * x_float).sum(0)
        if self.linear_terms is not None:
            scaled_energy += self.linear_terms @ x_float

        if not self.is_linear:
            scaled_energy *= self.scaling

        if self.constant_term is not None:
            scaled_energy += self.constant_term

        return scaled_energy

    def solve(self, xi, batch_size, max_iters, dt, init_noise):
        """
        使用给定参数运行指定的SB算法

        参数:
            xi: 耦合强度参数
            batch_size: 批处理大小
            max_iters: 最大迭代次数
            dt: 时间步长
            init_noise: 初始化噪声强度

        返回:
            最优能量值
        """
        # 初始化
        x = init_noise * (torch.randn((self.num_vars, batch_size), dtype=torch.float, device=self.device)) + 0.5
        y = init_noise * (torch.randn((self.num_vars, batch_size), dtype=torch.float, device=self.device))

        # 泵浦振幅
        pt = np.linspace(0, 1, max_iters)

        # 对于DSB，需要初始化x_sign
        if self.algorithm == "DSB":
            x_sign = torch.where(x > 0.5, 1.0, 0.0)

        # 主迭代循环
        for i in range(max_iters):
            # 根据算法类型计算梯度
            if self.algorithm == "BSB":
                # BSB: 使用连续变量x计算梯度
                grad = self.calc_grad(x)
            else:  # DSB
                # DSB: 使用二元变量x_sign计算梯度，每次迭代更新
                grad = self.calc_grad(x_sign)

            # 根据优化方向调整梯度符号
            if self.objective_sense == "maximize":
                grad_term = xi * grad
            else:
                grad_term = -xi * grad

            # 更新动量和位置
            y += (-(1 - pt[i]) * (x - 0.5) + grad_term) * dt
            x += y * dt

            # 边界条件处理
            cond = torch.abs(x - 0.5) > 0.5
            x = torch.where(cond, (torch.sign(x - 0.5) + 1) / 2.0, x)
            y = torch.where(cond, 0, y)

            # DSB特有：每次迭代更新x_sign
            if self.algorithm == "DSB":
                x_sign = torch.where(x > 0.5, 1.0, 0.0)

        # 计算所有批次的能量值并选择最优解
        best_value = float("-inf") if self.objective_sense == "maximize" else float("inf")

        for i in range(x.shape[1]):
            binary_solution = (x[:, i] > 0.5).int()
            current_value = self.calc_energy(binary_solution).item()

            if self.objective_sense == "maximize":
                if current_value > best_value:
                    best_value = current_value
            else:
                if current_value < best_value:
                    best_value = current_value

        return best_value


class OptunaOptimizer:
    def __init__(self, sb_optimizer, n_trials=100, fixed_sigma=None):
        """
        Optuna超参数优化器

        参数:
            sb_optimizer: SBOptimizer实例
            n_trials: 优化试验次数
            fixed_sigma: 固定的sigma值，如果为None则也会优化sigma
        """
        self.sb = sb_optimizer
        self.n_trials = n_trials
        self.best_params = None
        self.best_value = None
        self.fixed_sigma = fixed_sigma

    def objective(self, trial):
        """
        Optuna优化目标函数

        参数:
            trial: Optuna trial对象

        返回:
            目标函数值(对于minimize问题，返回正值；对于maximize问题，返回负值)
        """
        # 定义超参数搜索空间 - 调优 xi 和可选的 sigma
        xi_multiplier = trial.suggest_float("xi_multiplier", 0.1, 10.0, log=True)
        xi = self.sb.default_xi * xi_multiplier

        # sigma: 如果固定则使用固定值，否则也进行优化
        if self.fixed_sigma is not None:
            init_noise = self.fixed_sigma
        else:
            init_noise = trial.suggest_float("init_noise", 0.05, 0.3)

        # 固定其他参数
        batch_size = 100
        max_iters = 1000
        dt = 0.1

        # 运行指定的SB算法
        try:
            energy = self.sb.solve(xi, batch_size, max_iters, dt, init_noise)

            # 计算与目标值的gap(如果提供了目标值)
            if self.sb.target_value is not None:
                gap = abs(energy - self.sb.target_value)
                trial.set_user_attr("gap", gap)
                trial.set_user_attr("gap_percent", gap / abs(self.sb.target_value) * 100)

            trial.set_user_attr("energy", energy)
            trial.set_user_attr("xi", xi)

            # 返回值:
            # - 对于maximize问题: 返回负能量(optuna默认最小化)
            # - 对于minimize问题: 返回正能量
            if self.sb.objective_sense == "maximize":
                return -energy  # 转换为最小化问题
            else:
                return energy

        except Exception as e:
            print(f"❌ Trial失败: {e}")
            return float("inf")

    def optimize(self, study_name=None):
        """
        运行Optuna优化

        参数:
            study_name: study的名称

        返回:
            优化后的最佳参数字典
        """
        if study_name is None:
            study_name = f"{self.sb.algorithm}_{self.sb.problem_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print("\n" + "=" * 70)
        print(f"🚀 开始Optuna超参数优化 ({self.sb.algorithm}算法)")
        print("=" * 70)
        print(f"Study名称: {study_name}")
        print(f"优化试验次数: {self.n_trials}")
        print(f"优化方向: {'maximize' if self.sb.objective_sense == 'maximize' else 'minimize'}")
        print()

        # 创建study
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",  # 总是最小化(因为我们在objective中已经转换了)
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # 运行优化
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

        # 提取最佳结果
        best_trial = study.best_trial
        self.best_params = best_trial.params

        # 计算实际的最佳能量值
        if self.sb.objective_sense == "maximize":
            self.best_value = -best_trial.value  # 转换回原始值
        else:
            self.best_value = best_trial.value

        # 输出结果
        print("\n" + "=" * 70)
        print(f"✅ {self.sb.algorithm}算法优化完成!")
        print("=" * 70)
        print(f"\n🏆 最佳结果:")
        print(f"  - 目标函数值: {self.best_value:.6f}")
        if self.sb.target_value is not None:
            gap = abs(self.best_value - self.sb.target_value)
            gap_percent = gap / abs(self.sb.target_value) * 100
            print(f"  - 与最优值的差距: {gap:.6f} ({gap_percent:.2f}%)")

        print(f"\n📋 最佳超参数:")
        xi_best = self.sb.default_xi * self.best_params["xi_multiplier"]
        print(f"  - xi: {xi_best:.6f} (默认值的 {self.best_params['xi_multiplier']:.2f} 倍)")

        if self.fixed_sigma is not None:
            print(f"  - init_noise (sigma): {self.fixed_sigma:.2f} (固定)")
        else:
            print(f"  - init_noise (sigma): {self.best_params.get('init_noise', 0.1):.2f} (优化)")

        print(f"  - batch_size: 100 (固定)")
        print(f"  - max_iters: 1000 (固定)")
        print(f"  - dt: 0.1 (固定)")

        # 保存结果
        result_dict = {
            "algorithm": self.sb.algorithm,
            "problem_name": self.sb.problem_name,
            "objective_sense": self.sb.objective_sense,
            "target_value": self.sb.target_value,
            "best_value": self.best_value,
            "best_params": {
                "xi": xi_best,
                "xi_multiplier": self.best_params["xi_multiplier"],
                "batch_size": 100,
                "max_iters": 1000,
                "dt": 0.1,
                "init_noise": (
                    self.fixed_sigma if self.fixed_sigma is not None else self.best_params.get("init_noise", 0.1)
                ),
            },
            "default_xi": self.sb.default_xi,
            "num_vars": self.sb.num_vars,
            "fixed_sigma": self.fixed_sigma,
            "timestamp": datetime.now().isoformat(),
        }

        if self.sb.target_value is not None:
            result_dict["gap"] = gap
            result_dict["gap_percent"] = gap_percent

        # 保存为JSON
        if self.fixed_sigma is not None:
            output_file = (
                Path("save_json")
                / f"optuna_results_{self.sb.algorithm}_{self.sb.problem_name}_sigma{self.fixed_sigma:.2f}.json"
            )
        else:
            output_file = (
                Path("save_json") / f"optuna_results_{self.sb.algorithm}_{self.sb.problem_name}_varyingSigma.json"
            )

        # 确保目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"\n💾 结果已保存到: {output_file}")

        # 显示前10个最佳trials
        print(f"\n📊 前10个最佳试验:")
        print("-" * 70)
        print(f"{'Trial':<8} {'Value':<12} {'xi_mult':<10} {'xi':<12}")
        print("-" * 70)

        sorted_trials = sorted(study.trials, key=lambda t: t.value)[:10]
        for trial in sorted_trials:
            value = -trial.value if self.sb.objective_sense == "maximize" else trial.value
            xi_val = self.sb.default_xi * trial.params["xi_multiplier"]
            print(f"{trial.number:<8} {value:<12.4f} {trial.params['xi_multiplier']:<10.4f} " f"{xi_val:<12.6f}")

        return result_dict


if __name__ == "__main__":
    import sys

    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✓ CUDA设备可用: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA版本: {torch.version.cuda}")
        device = "cuda"
    else:
        print("⚠ CUDA不可用，使用CPU")
        device = "cpu"

    print()

    # 配置优化任务
    QPLIB_FILE = "qplib/QPLIB_3565.qplib"  # QPLIB_3565问题
    TARGET_VALUE = 282  # 已知的最优值为282

    # 从命令行参数获取配置
    # 用法: python SB_optuna.py [n_trials] [algorithm] [sigma]
    # 示例: python SB_optuna.py 200 BSB 0.1
    # 示例: python SB_optuna.py 200 DSB 0.2
    # 示例: python SB_optuna.py 200 BSB auto  (自动优化sigma)

    if len(sys.argv) > 1:
        N_TRIALS = int(sys.argv[1])
    else:
        N_TRIALS = 50  # 默认值

    if len(sys.argv) > 2:
        ALGORITHM = sys.argv[2].upper()
        if ALGORITHM not in ["BSB", "DSB"]:
            print(f"❌ 不支持的算法类型: {sys.argv[2]}")
            print("支持的算法: BSB, DSB")
            sys.exit(1)
    else:
        # 交互式选择算法
        print("=" * 70)
        print("请选择要优化的算法:")
        print("  1. BSB - 使用连续变量计算梯度")
        print("  2. DSB - 使用二元变量计算梯度，每次迭代更新")
        print("=" * 70)
        choice = input("\n请输入选项 (1/2, 默认1): ").strip()

        if choice == "2":
            ALGORITHM = "DSB"
        else:
            ALGORITHM = "BSB"

    # 获取 sigma 参数
    if len(sys.argv) > 3:
        sigma_arg = sys.argv[3].lower()
        if sigma_arg == "auto":
            FIXED_SIGMA = None  # 自动优化 sigma
        else:
            try:
                FIXED_SIGMA = float(sys.argv[3])
            except ValueError:
                print(f"❌ 无效的sigma值: {sys.argv[3]}")
                print("sigma应该是数字(如0.1)或'auto'")
                sys.exit(1)
    else:
        # 交互式选择 sigma
        print("\n" + "=" * 70)
        print("请选择 sigma (初始化噪声) 的处理方式:")
        print("  1. 固定 sigma = 0.05")
        print("  2. 固定 sigma = 0.1 (推荐)")
        print("  3. 固定 sigma = 0.15")
        print("  4. 固定 sigma = 0.2")
        print("  5. 固定 sigma = 0.3")
        print("  6. 自动优化 sigma (与xi一起优化)")
        print("=" * 70)
        sigma_choice = input("\n请输入选项 (1-6, 默认2): ").strip()

        sigma_map = {"1": 0.05, "2": 0.1, "3": 0.15, "4": 0.2, "5": 0.3, "6": None}
        FIXED_SIGMA = sigma_map.get(sigma_choice, 0.1)

    print()
    print("=" * 70)
    print("=" * 70)
    print(f"配置信息:")
    print(f"  - 算法: {ALGORITHM}")
    print(f"  - 试验次数: {N_TRIALS}")
    print(f"  - 目标值: {TARGET_VALUE}")
    if FIXED_SIGMA is not None:
        print(f"  - Sigma: {FIXED_SIGMA} (固定)")
    else:
        print(f"  - Sigma: 自动优化 [0.05, 0.3]")
    print("=" * 70)
    print()

    # 创建SB优化器
    sb_opt = SBOptimizer(QPLIB_FILE, device=device, target_value=TARGET_VALUE, algorithm=ALGORITHM)

    # 创建Optuna优化器并运行
    optuna_opt = OptunaOptimizer(sb_opt, n_trials=N_TRIALS, fixed_sigma=FIXED_SIGMA)
    results = optuna_opt.optimize()

    print("\n" + "=" * 70)
    print("🎉 所有优化任务完成!")
    print("=" * 70)
