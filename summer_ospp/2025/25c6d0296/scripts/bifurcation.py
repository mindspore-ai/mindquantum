import os
import json
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu, svds
import itertools
import multiprocessing
from collections import deque

class HuboProblem:
    """
    用一个可被Pickle的类来封装HUBO问题的定义。
    """

    def __init__(self, M_rows, N_cols, A):
        self.M_rows = M_rows
        self.N_cols = N_cols
        self.A = A

        num_h = (self.M_rows + 1) * self.N_cols
        self.n_vars = num_h + self.M_rows * (self.N_cols + 1)

        self.faces = []
        for r in range(self.M_rows):
            for c in range(self.N_cols):
                self.faces.append((
                    r * self.N_cols + c,
                    (r + 1) * self.N_cols + c,
                    num_h + r * (self.N_cols + 1) + c,
                    num_h + r * (self.N_cols + 1) + c + 1
                ))
        print(f"构建网格HUBO问题: {self.M_rows}x{self.N_cols}, 变量数: {self.n_vars}")

    def energy(self, x):
        """
        计算给定状态向量 x 的 HUBO 能量 H(x)。
        """
        # 计算四体相互作用项
        hubo_energy = 0
        for i, j, k, l in self.faces:
            hubo_energy -= x[i] * x[j] * x[k] * x[l]
        # 计算惩罚项
        penalty_energy = self.A * np.sum((x**2 - 1)**2)
        return hubo_energy + penalty_energy


    def F(self, x):
        """计算能量函数的梯度 F(x)"""
        hubo_grad = np.zeros(self.n_vars)
        for i, j, k, l in self.faces:
            p = x[i] * x[j] * x[k] * x[l]
            denom_i = x[i] if x[i] != 0 else 1e-12
            denom_j = x[j] if x[j] != 0 else 1e-12
            denom_k = x[k] if x[k] != 0 else 1e-12
            denom_l = x[l] if x[l] != 0 else 1e-12
            hubo_grad[i] -= p / denom_i
            hubo_grad[j] -= p / denom_j
            hubo_grad[k] -= p / denom_k
            hubo_grad[l] -= p / denom_l
        return hubo_grad + 4 * self.A * x * (x ** 2 - 1)

    def nabla_F(self, x):
        """计算 F(x) 的雅可比矩阵 ∇F(x)"""
        J_hubo = sp.lil_matrix((self.n_vars, self.n_vars))
        for i, j, k, l in self.faces:
            face = (i, j, k, l)
            for u, v in itertools.combinations(face, 2):
                p, q = [idx for idx in face if idx != u and idx != v]
                coupling = -x[p] * x[q]
                J_hubo[u, v] += coupling
                J_hubo[v, u] += coupling
        return sp.diags(4 * self.A * (3 * x ** 2 - 1), format='csc') + J_hubo.tocsc()

class ArcLengthSolver:
    def __init__(self, problem_obj, z_start, ds, max_steps, branch_id, parent_id, parent_bif_point=None):
        self.problem = problem_obj  # <--- 修改: 接收 problem 对象
        self.z = np.array(z_start, dtype=float)
        self.ds = ds
        # ... (其余 __init__ 代码不变) ...
        self.max_steps = max_steps
        self.branch_id = branch_id
        self.parent_id = parent_id
        self.n_vars = len(z_start) - 1
        self.path = [parent_bif_point, self.z.copy()] if parent_bif_point is not None else [self.z.copy()]
        self.tangent = None
        self.step_count = 0
        self.last_result = {}
        self.x0 = z_start[:-1]
        # <--- 修改: 调用 problem 对象的方法
        self.F_homotopy = lambda z: (1 - z[-1]) * (z[:-1] - self.x0) + z[-1] * self.problem.F(z[:-1])
        # ds调整策略需要的变量
        self.p_dir = 1
        self.cooldown = 0
        #
        self.d_toler = 0.001
        self.n_toler = 5
        self.max_p_deque = deque(maxlen=self.n_toler)
        self.n_stuck = 0
        self.last_ds = 0

    def _get_derivatives(self, z):
        # 计算同伦函数 H(x, p) 在当前点 z = (x, p) 处，分别对解向量 x 和同伦参数 p 的偏导数
        x, p = z[:-1], z[-1]
        nabla_F = self.problem.nabla_F(x)
        H_x = (1 - p) * sp.identity(self.n_vars, format='csc') + p * nabla_F
        F_val = self.problem.F(x)
        H_p = -(x - self.x0) + F_val
        return H_x, H_p

    def _compute_tangent(self):
        H_x, H_p = self._get_derivatives(self.z)  # 计算出当前点 z=(x, p) 的两个关键偏导数矩阵：H_x 和 H_p
        try:
            x_dot = spsolve(H_x, -H_p)
        except RuntimeError:
            return None, H_x
        tangent = np.append(x_dot, 1.0)  # 路径的单位方向向量
        tangent /= np.linalg.norm(tangent)
        if self.tangent is not None and np.dot(tangent, self.tangent) < 0:
            tangent *= -1
        return tangent, H_x

    def _corrector(self, z_pred, tangent):
        z_k = z_pred.copy()
        for i in range(1, 11):  # 循环中z_k表示当前猜测坐标
            H_x, H_p = self._get_derivatives(z_k)  # x和p分别移动一点，对同伦函数H(x, p)造成的影响
            # 计算方程组整体对变量 z 的“总敏感度”，也就是整个系统的雅可比矩阵J_aug
            J_aug = sp.vstack([sp.hstack([H_x, sp.csc_matrix(H_p[:, np.newaxis])]), sp.csr_matrix(tangent)]).tocsc()
            # 计算当前猜测点 z_k 的“总误差”
            R = np.append(self.F_homotopy(z_k), np.dot(z_k - z_pred, tangent))
            if np.linalg.norm(R) < 1e-8: return z_k, i # 如果误差足够小，说明猜测正确，这个点回归成功，返回这个坐标
            try:
                delta_z = splu(J_aug).solve(-R)  # 计算修正量delta_z
            except (RuntimeError, np.linalg.LinAlgError):
                return None, i
            z_k += delta_z  # 把修正量delta_z叠加到z_k上
        return None, i

    @staticmethod
    def _analyze_singularity(H):
        # return: 找到的所有方向向量
        try:
            k = min(3, H.shape[0] - 2)  # k是我们要找的奇异值数量,最多3个
            if k <= 0: return []
            # 调用svds找到幅度最小 (Smallest Magnitude)的奇异值,
            # s是包含奇异值的np数组, 形状(k,)
            # vt是每个奇异值对应的右奇异向量，形状(k, n_vars)
            _, s, vt = svds(H, k=k, which='SM')
            s_max = svds(H, k=1, which='LM')[1][0] # 再找到幅度最大的奇异值，用于判断上面的奇异值够不够小
            return [v.T for s_val, v in zip(s, vt) if s_val / s_max < 1e-6]  # 筛选出足够小的奇异值对应的向量，转置为列向量并返回
        except Exception:
            return []


    def step(self, i_solver):
        # 超出step就停止
        if self.step_count >= self.max_steps:
            print(f'ArcLengthSolver{i_solver}: self.step_count surpass {self.max_steps}, return TERMINATED')
            self.last_result = {'status': 'TERMINATED', 'reason': 'Max steps reached'}
            return self.last_result
        self.step_count += 1

        self.last_ds = self.ds
        p_old = self.z[-1]

        self.tangent, H_x_old = self._compute_tangent()  # 获得方向向量(包含p和s) self.tangent

        if self.tangent is None: # 如果求不出斜率,或者当前方向p分量很小,说明可能是到了拐点/分岔点
            v_list = self._analyze_singularity(H_x_old)  # 计算是否存在足够小的奇异值，并返回这些奇异值对应的向量
            bp_data = {'z': self.z, 'p': self.z[-1], 'v_list': v_list}
            self.last_result = {'status': 'BIFURCATION', 'data': bp_data}
            if self.tangent is None:
                print(f'ArcLengthSolver{i_solver}:tangent is None, v length {len(v_list)}, BIFURCATION')
            else:
                print(f'ArcLengthSolver{i_solver}:p tangent is small, v length {len(v_list)}, BIFURCATION')
            return self.last_result

        z_pred = self.z + self.ds * self.tangent  # 前面的if没触发，说明此时不是拐点，我们照着当前方向self.tangent走一步
        z_new, iters = self._corrector(z_pred, self.tangent)  # 再根据corrector纠正回来,如果成功会得到正确的z_new，否则z_new为None
        if z_new is None:
            v_list = self._analyze_singularity(H_x_old)  # corrector失败了，需要先检查是否是分岔点
            if v_list: # 找到了奇异向量，这就是一个分岔点
                print(f"ArcLengthSolver{i_solver}: _corrector failed and bifurcation found, {len(v_list)} vectors.")
                bp_data = {'z': self.z, 'p': self.z[-1], 'v_list': v_list}
                self.last_result = {'status': 'BIFURCATION', 'data': bp_data}
                return self.last_result
            else:
                self.ds /= 2
                reason = 'Corrector failed'
                status = 'RETRYING' if self.ds > 1e-9 else 'TERMINATED'
                if status == 'RETRYING': print(f'ArcLengthSolver{i_solver}: _corrector failed, ds={self.ds}, RETRYING')
                else: print(f'ArcLengthSolver{i_solver}: _corrector failed, ds={self.ds} too small, TERMINATED')
                self.last_result = {'status': status, 'reason': reason}
                return self.last_result

        # 继续前进
        self.z = z_new
        self.path.append(self.z.copy())
        p_new = self.z[-1]

        if p_new < p_old:
            self.max_p_deque.append(p_new)
            if len(self.max_p_deque) == self.max_p_deque.maxlen and max(self.max_p_deque) - min(self.max_p_deque) < self.d_toler:
                print(f"ArcSolver {i_solver}: p={p_new} for {self.n_stuck} steps. BIFURCATION.")
                # 注意：我们这里不需要v_list，因为我们不依赖它来找方向
                self.last_result = {'status': 'BIFURCATION', 'data': {'z': self.z, 'p': p_new, 'v_list': []}}
                return self.last_result

        print(f'ArcLengthSolver{i_solver}: tangent exists, move on, p={p_new}, ds={self.ds}')

        self._adapt_ds(p_new, p_old, iters)

        self.last_result = {'status': 'CONTINUING'}
        return self.last_result

    def _adapt_ds(self, p_new, p_old, iters):
        """根据当前步的结果，智能调整下一步的步长ds"""
        MAX_DS = 0.05
        COOLDOWN_STEPS = 10

        # 1. 如果在冷静期，计数器减1
        if self.cooldown > 0:
            self.cooldown -= 1

        # 2. 检测p的方向是否反转
        p_dir = np.sign(p_new - p_old)
        if p_dir == 0: p_dir = self.p_dir # p值不变则方向不变

        is_turn = (self.p_dir != 0 and p_dir != self.p_dir)

        # 3. 更新p的方向以备下一步使用
        if p_dir != 0:
            self.p_dir = p_dir

        # 4. 根据是否掉头、是否在冷静期，来决定ds调整策略
        if is_turn:
            # 检测到掉头：猛踩刹车，并进入冷静期
            print(f"ArcSolver: Turn detected! Reducing ds and entering cooldown.")
            self.ds /= 4
            self.cooldown = COOLDOWN_STEPS
        else:
            # 正常行驶
            if self.cooldown > 0:
                # 在冷静期：缓慢恢复速度
                self.ds *= 1.05
            else:
                # 不在冷静期：根据收敛速度常规驾驶
                if iters < 4:
                    self.ds *= 1.2  # 油门
                elif iters > 7:
                    self.ds *= 0.75 # 刹车

        # 确保ds不超过上限
        self.ds = min(self.ds, MAX_DS)

class BifurcationManager:
    def __init__(self, problem_obj, x0, **params):
        self.problem = problem_obj
        self.params = params
        self.bifurcation_points = []
        self.completed_paths = []
        self.path_counter = 1
        z_start = np.append(x0, 0.0)
        initial_solver = ArcLengthSolver(problem_obj, z_start, params['ds'], params['max_solver_steps'], 0, -1)
        self.active_solvers = [initial_solver]

    def run(self, max_bifurcations=5):
        steps = 0
        max_total_steps = self.params.get('max_total_steps', 5000)
        num_bifurcation = 0
        reach_bifurcation_upbound = False
        while self.active_solvers and steps < max_total_steps:
            print(f'----- Step {steps} start -----'); steps += 1
            if len(self.bifurcation_points) >= max_bifurcations:
                break
            newly_spawned, solvers_to_remove = [], []

            print(f'Active solvers: {len(self.active_solvers)}, Bifurcations: {num_bifurcation}')
            for i_solver, solver in enumerate(self.active_solvers):
                if reach_bifurcation_upbound: break
                result = solver.step(i_solver)
                if result['status'] == 'BIFURCATION':
                    solvers_to_remove.append(solver)
                    bp_data = result['data']
                    self.bifurcation_points.append(bp_data)
                    if bp_data.get('v_list'):
                        if len(self.active_solvers) < self.params.get('max_active_paths', 10):
                            newly_spawned.extend(self._spawn_new_paths(solver, bp_data['z'], bp_data['v_list'], solver.tangent))
                    else:  # 进入我们自己确定方向的方法
                        z_star = result['data']['z']
                        found_paths = self._find_new_directions(solver, z_star)
                        # 3. 为每一条被验证过的新路，派出新的Solver
                        for z_corrected, T_new in found_paths:
                            if len(self.active_solvers) + len(newly_spawned) >= self.params.get('max_active_paths', 10):
                                break  # 控制并发数量
                            # 创建新Solver，注意起点是z_corrected
                            new_solver = ArcLengthSolver(self.problem, z_corrected, self.params['ds'],
                                                         self.params['max_solver_steps'], self.path_counter,
                                                         solver.branch_id, parent_bif_point=z_star)
                            self.path_counter += 1
                            newly_spawned.append(new_solver)
                    print(f'Bifurcations: {num_bifurcation}')
                    num_bifurcation += 1
                    if num_bifurcation >= max_bifurcations:
                        reach_bifurcation_upbound = True
                        break
                elif result['status'] in ['TERMINATED', 'RETRYING']:
                    solvers_to_remove.append(solver)
            self._update_solvers(solvers_to_remove, newly_spawned)
        self._finalize_paths()
        return self.completed_paths


    def _find_new_directions(self, solver_old, z_star):
        """
        通过循环随机尝试，从一个点z_star寻找最多2条新的、有效的、不同的路径。

        Args:
            solver_old: 卡住的那个旧的solver实例。
            z_star: 卡住的点坐标。

        Returns:
            一个列表，包含找到的新路径的元组 (z_corrected, tangent_new)
        """
        print(f"Manager: Starting trial-and-error search for 2 new directions from p={z_star[-1]:.6f}")

        T_old = solver_old.tangent
        if T_old is None:  # 如果旧的切线不存在，无法比较，直接返回
            return []

        # 归一化旧的切线方向（只关心x分量）
        T_old_x_norm = T_old[:-1] / np.linalg.norm(T_old[:-1])
        n_vars = self.problem.n_vars

        found_paths = []
        max_trials = 200  # 设置一个尝试上限，防止死循环

        for i in range(max_trials):
            if len(found_paths) >= 2:
                break  # 已经找到两个了，任务完成

            # 1. 生成一个随机的候选方向 (只生成x分量)
            T_candidate_x = np.random.randn(n_vars)
            T_candidate_x /= np.linalg.norm(T_candidate_x)

            # 2. 安检1：和老路的方向是否太相似？
            if abs(np.dot(T_candidate_x, T_old_x_norm)) > 0.98:
                continue  # 太像了，跳过

            # 3. 安检2：和已找到的新路是否太相似？
            is_similar_to_found = False
            for _, T_found in found_paths:
                T_found_x_norm = T_found[:-1] / np.linalg.norm(T_found[:-1])
                if abs(np.dot(T_candidate_x, T_found_x_norm)) > 0.95:
                    is_similar_to_found = True
                    break
            if is_similar_to_found:
                continue  # 和已找到的太像了，跳过

            # 4. 安检3：验证这个方向是否“可通行”？
            #    我们假设新方向在p维度上是平的，先迈一小步
            T_candidate_aug = np.append(T_candidate_x, 0.0)

            test_step = solver_old.ds if solver_old.ds > 1e-4 else self.params['ds'] * 0.1
            z_pred = z_star + test_step * T_candidate_aug

            # 使用旧solver的corrector来测试
            z_corrected, iters = solver_old._corrector(z_pred, T_candidate_aug)

            if z_corrected is not None:
                # 成功了！
                p_new = z_corrected[-1]
                print(f"  --> Trial #{i + 1}: SUCCESS! Found a valid path at p={p_new:.6f}")

                # 从这个修正后的点，重新计算一个更准确的切线作为最终方向
                temp_solver = ArcLengthSolver(self.problem, z_corrected, self.params['ds'], 1, 0, 0)
                T_new, _ = temp_solver._compute_tangent()

                if T_new is not None:
                    found_paths.append((z_corrected, T_new))

        print(f"Manager: Search finished. Found {len(found_paths)} new paths after {i + 1} trials.")
        return found_paths

    def _spawn_new_paths(self, parent_solver, z_star, v_list, parent_tangent):
        spawned = []
        if parent_tangent is None:
            return []
        parent_tangent_x = parent_tangent[:-1] / np.linalg.norm(parent_tangent[:-1])

        orthogonality_threshold = 0.5
        for v in v_list:
            for v in v_list:
                v_normalized = v / np.linalg.norm(v)
                dot_product = np.abs(np.dot(parent_tangent_x, v_normalized))  # 计算父路径方向(x分量)与新零向量方向的点积
                # 如果点积很大（接近1），说明方向几乎相同，这是个拐点，跳过。
                if dot_product > orthogonality_threshold:
                    # print(f"DEBUG: Detected a turning point for branch {parent_solver.branch_id}. Skipping spawn.")
                    continue

            for sign in [1, -1]:
                tangent_guess = np.append(sign * v, 0.0)
                z_pred = z_star + parent_solver.ds * 0.1 * tangent_guess
                z_branch_start, corrector_iters = parent_solver._corrector(z_pred, tangent_guess)
                if z_branch_start is not None:
                    new_solver = ArcLengthSolver(self.problem, z_branch_start, self.params['ds'],
                                                 self.params['max_solver_steps'], self.path_counter,
                                                 parent_solver.branch_id, parent_bif_point=z_star)
                    self.path_counter += 1
                    spawned.append(new_solver)
        return spawned

    def _update_solvers(self, to_remove, to_add):
        for s in set(to_remove):
            self.completed_paths.append(self._create_record(s))
            if s in self.active_solvers:
                self.active_solvers.remove(s)
        self.active_solvers.extend(to_add)

    def _finalize_paths(self):
        for s in self.active_solvers:
            self.completed_paths.append(self._create_record(s))
        self.active_solvers = []

    @staticmethod
    def _create_record(solver):
        return {'branch_id': solver.branch_id,
                'parent_id': solver.parent_id,
                'path_data': np.array(solver.path),
                'status': solver.last_result.get('status', 'ACTIVE'),
                'bifurcation_info': solver.last_result.get('data') if solver.last_result.get(
                    'status') == 'BIFURCATION' else None}

def analyze_and_report_results(records, faces):
    print("\n" + "=" * 60 + "\n最终结果分析\n" + "=" * 60)
    successful_paths = [r for r in records if r['path_data'].size > 0 and r['path_data'][-1, -1] >= 1.0]
    if not successful_paths:
        print("没有路径成功到达 p=1.0。")
        return None
    print(f"发现 {len(successful_paths)} 条成功到达 p=1.0 的路径。")
    best_energy, best_branch_id = np.inf, -1
    for record in successful_paths:
        x_final = record['path_data'][-1, :-1]
        s = np.sign(x_final)
        s[s == 0] = 1
        energy = 0
        for i, j, k, l in faces:
            energy -= s[i] * s[j] * s[k] * s[l]
        if energy < best_energy:
            best_energy = energy
            best_branch_id = record['branch_id']
        print(f"  -> 分支 #{record['branch_id']}: 最终HUBO能量 = {energy}")
    print("-" * 60 + f"\n最佳解来自 分支 #{best_branch_id}，最低能量: {best_energy}\n" + "=" * 60)
    return best_energy

def run_single_trial(args):
    """单个并行工作单元。"""
    trial_index, problem_obj, solver_params = args

    np.random.seed(int(time.time()) + trial_index)
    x_initial = np.random.randn(problem_obj.n_vars)

    manager = BifurcationManager(problem_obj, x_initial, **solver_params)
    final_records = manager.run(max_bifurcations=5)

    if len(manager.bifurcation_points) >= 5:
        # 为了避免传递大量数据，只返回必要的信息
        return {"status": "Success", "trial_index": trial_index, "records": final_records}
    else:
        return {"status": "Failure", "trial_index": trial_index}

class NumpyArrayEncoder(json.JSONEncoder):
    """
    自定义一个JSON编码器，用于处理Numpy数组。
    它会检查对象是否是np.ndarray类型，如果是，就调用其tolist()方法转换成列表。
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)