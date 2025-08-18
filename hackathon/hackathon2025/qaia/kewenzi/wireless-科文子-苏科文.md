# 基于量子启发算法的高效波束成形优化：联合振幅与相位设计在MIMO中的应用

队员：苏科文 西安电子科技大学  指导老师：苏兆锋 中国科学技术大学

## 摘要

量子启发算法作为一种经典计算与量子思想结合的产物，在阵列规模较大的情况下有重要应用价值。本文基于量子启发算法，联合振幅与相位处理，实现了高效的波束赋形优化。

针对大规模天线阵列（Massive MIMO）系统中传统波束赋形优化方法面临的高计算复杂度与低收敛效率问题，本文提出了一种融合量子启发算法与混合变量优化的动态波束成形框架，通过分阶段优化流程实现振幅与相位的协同设计。该方法首先利用量子启发的模拟分叉（BSB）算法优化二进制相位编码，最小化旁瓣能量与主瓣能量比值；随后基于Adam优化器调整振幅分布，进一步压缩主瓣宽度并抑制旁瓣；最终引入混合变量遗传算法（MixedVariableGA）动态调整多段旁瓣压制区域权重与方向约束，同步优化实数振幅与二进制相位，并通过动态压制区域机制、物理约束强化及混合优化目标函数提升算法对复杂干扰场景的适应性与物理可行性。实验验证表明，在32单元线性阵列场景下，本文方法最优测试得分达419.8216，主瓣宽度稳定于  $6^{\circ}$  且方向偏差控制在  $1^{\circ}$  以内，为5G/6G大规模MIMO系统提供了低复杂度、高鲁棒性的优化解决方案。

**关键词：量子启发算法、混合变量优化、BSB、GA。**

## 1 问题背景与描述

### 1.1 相关背景

波束赋形是无线通信领域的关键技术，通过调整天线阵列中各阵元的相位和振幅，可以集中信号能量、提升通信质量和容量。随着5G及下一代无线通信的发展，大规模天线阵列（Massive MIMO）技术日益重要。然而，传统优化方法在处理大规模阵列时面临计算复杂度爆炸的问题。量子启发算法提供了一种高效解决方案，通过模拟量子退火或量子比特行为，在经典计算机上实现近似量子优化。这类算法在组合优化问题中展现了

快速收敛和高解质量的优势，尤其适用于波束赋形这类涉及离散变量的复杂场景。研究显示，量子启发的模拟分叉（SB）算法相比遗传算法具有更快的速度，且在处理多波束和旁瓣抑制等复杂场景时表现优异。

### 1.2 赛题内容

通过调整天线阵列中天线阵子的振幅和相位可以实现针对特定场景的波束赋形，达成收窄主瓣的波束宽度、压制特定区域旁瓣的信号强度等目的。该问题是典型的组合优化问题，量子启发算法在处理组合优化问题上具有独特优势。

本赛题基于理想天线建模，构建等间距半波长线阵，希望选手基于量子启发算法，实现主瓣方向一定连续范围内最优相位一振幅序列最优解的自动搜寻，使成形方向得到压制，旁瓣强度低于主瓣方向的- 15dB，主瓣宽度减小，从而使波束赋形场景得以优化。

<div align="center">
  <img src="https://cdn-mineru.openxlab.org.cn/result/2025-08-04/5396361b-55f4-4fbb-a3b6-a212ef42b5f5/64d5ec4eadacf47cf57ddc55a2afbb4673903b191a787cbb58f015afc501f39a.jpg" width="400" />
  <p>图1 理想天线线阵模型示意图</p>
</div>

关系式：

$$
|F(\theta)|^2 = E(\theta)\left[\sum_{n = 1}^{N}A_n(\theta)\right]\left[\sum_{n = 1}^{N}A_n^* (\theta)\right] \tag{1}
$$

阵因子：

$$
A_{n}(\theta) = I_{n}\exp \{\pi in\cos \theta \}^{+} \tag{2}
$$

天线单元因子：

$$
E(\theta) = 10\frac{E_{dB}(\theta)}{10} \tag{3}
$$

$$
E_{dB}(\theta) = -\min \left\{12\left(\frac{\theta - 90^\circ}{90^\circ}\right)^2,30\right\} \tag{4}
$$

## 2 问题分析

### 2.1 研究现状

波束赋形（Beamforming,BF）技术作为提升无线通信系统性能的关键手段，一直受到学术界和工业界的广泛关注。近年来，随着大规模MIMO（Massive MIMO）技术的兴起，如何高效地优化波束赋形成为研究热点。大规模MIMO通过在基站部署大量天线，能够显著提升信号质量、扩展覆盖范围和增加系统容量。然而，天线阵列规模的增大也带来了优化问题的复杂性呈指数级增长的挑战。传统优化方法如数字波束赋形（Digital BF）、全息波束赋形（Holographic BF）和遗传算法（Genetic Algorithms）在处理大规模阵列时，往往面临计算效率低和容易陷入局部最优解的问题。

为了应对这些挑战，量子启发的优化算法逐渐崭露头角。其中，模拟分叉（Simulated Bifurcation,SB）算法因其高效的计算能力和出色的解质量而备受关注。SB算法通过模拟非线性哈密顿系统中的分叉现象，能够在经典计算机上实现类似量子退火的过程，从而快速找到组合优化问题的近似解。与传统方法相比，SB算法在优化时间和解的质量上均表现出显著优势。例如，SB算法的优化时间比遗传算法快100倍以上，且在处理复杂场景如单波束旁瓣抑制和多波束零点控制时表现尤为出色。

如图2所示：

<div align="center">
  <img src="https://cdn-mineru.openxlab.org.cn/result/2025-08-04/5396361b-55f4-4fbb-a3b6-a212ef42b5f5/ba74cdfa95596494e267ad800a1fd3cb7807765e8d39ef940b4aecdbc2dad625.jpg" width="500"/>
  <p>图2 现状分析过程示意图</p>
</div>

### 2.2模拟分叉（SimulatedBifurcation，SB）算法

模拟分岔算法（SimulatedBifurcationAlgorithm）可以为复杂的大规模组合优化问题获得精确的结果。这种算法利用了分岔现象：经典力学中的绝热过程和遍历过程，使得复杂的行为能够被抽象成简单且的模拟数值。

如图3所示：

<div align="center">
  <img src="https://cdn-mineru.openxlab.org.cn/result/2025-08-04/5396361b-55f4-4fbb-a3b6-a212ef42b5f5/c8be2ac1648d2796bcc3429d42f45aab5069105e16ab39139de90606180859ab.jpg" width="500"/>
  <p>图3 模拟分叉现象示意图</p>
</div>

### 2.3符号说明

表1符号说明表  

<table><tr><td>符号</td><td>说明</td><td>单位</td></tr><tr><td>θ0</td><td>主瓣方向角</td><td>度</td></tr><tr><td>N</td><td>天线阵列的阵元数量
（本文为32）</td><td>个</td></tr><tr><td>encode_qubit</td><td>相位编码比特数</td><td>\</td></tr><tr><td>xi</td><td>优化步长权重超参数</td><td>\</td></tr><tr><td>weight</td><td>损失函数核心参数</td><td>\</td></tr><tr><td>F(θ)</td><td>波束方向图函数</td><td>\</td></tr><tr><td>E(θ)</td><td>天线单元因子</td><td>\</td></tr><tr><td>An(θ)</td><td>阵因子</td><td>\</td></tr><tr><td>phase_angle</td><td>阵元相位角</td><td>度</td></tr><tr><td>S</td><td>一定范围内旁瓣信号强度的加权平
均值</td><td>\</td></tr><tr><td>C</td><td>优化损失函数</td><td>\</td></tr><tr><td>x_bit</td><td>二进制解</td><td>\</td></tr></table>

## 3方案描述

本方案描述本赛题分为两种情形，即“是否优化振幅”：False代表不优化振幅；True代表优化振幅。接下来进行分开描述：

### 3.1不优化振幅，False的情形

方法选择：内外双层优化

内层使用dSB作为相位优化核心；外层使用遗传算法进行超参数（xi,weight等）优化。

<div align="center">
  <img src="https://cdn-mineru.openxlab.org.cn/result/2025-08-04/5396361b-55f4-4fbb-a3b6-a212ef42b5f5/85f25cab82d4f91def6b2dc889481969fa8e092a63d5eaaf5379e6f996d3f66d.jpg" width="500"/>
  <p>图4 内外双层优化示意图</p>
</div>

#### 3.1.1内层相位dSB优化

程序中采用的损失函数形式是最大化主瓣减去加权旁瓣的形式，并且将该形式转化为QAIA模块中接受的耦合矩阵J的输入形式。

$$
C = |F(\boldsymbol {\theta}_0)|^2 -wS \tag{5}
$$

其中，S是一定范围内旁瓣信号强度的加权平均值：

$$
S = a\nu g\{a_i|F(\theta_i)|^2\} ,\theta_i\in \left[0^\circ ,\theta_0 - \frac{W}{2}\right]\cup \left[\theta_0 + \frac{W}{2},180^\circ \right] \tag{6}
$$

具体代码如下：

```python

def opt_phase_QUBO(self):
    c1 = 0.5 + 0.5j
    c2 = 0.5 - 0.5j
    factor_array = torch.cat((self.efield[:, self._get_index(self.param['theta_0'])] * c1,
                              self.efield[:, self._get_index(self.param['theta_0'])] * c2), dim=0)
    J_enhance = torch.einsum('i, j -> ij', factor_array.conj(), factor_array).real

    J_suppress = 0.0
    for i, (start, end) in enumerate(self.param['range_list']):
        num = 0
        a_0 = 0.0
        max_energy = torch.zeros((64, 64))
        for j in range(round(self._get_index(self.param['theta_0'] + start)),
                       round(self._get_index(self.param['theta_0'] + end)), 1):
            num += 1
            factor_array = torch.cat((self.efield[:, j] * c1, self.efield[:, j] * c2), dim=0)
            a_0 += torch.einsum('i, j -> ij', factor_array.conj(), factor_array)

            current_energy = torch.einsum('i, j -> ij', factor_array.conj(), factor_array)
            norm_a = torch.norm(max_energy, p='fro')
            norm_b = torch.norm(current_energy, p='fro')
            max_energy = max_energy if norm_a > norm_b else current_energy

        J_suppress += self.param['range_list_weight'][i] * a_0 / num

    J = torch.real((self.param['weight'] * J_enhance - (1 - self.param['weight']) * J_suppress)).numpy()
    solver = BSB(np.array(J, dtype="float64"), batch_size=5, n_iter=self.param['n_iter'],
                 xi=self.param['xi'], backend='cpu-float32')
    solver.update()

    array = solver.x
    min_energy_index = np.argmin(solver.calc_energy())
    max_cut_index = np.argmax(solver.calc_cut())
    solution_max_cut = array[:, max_cut_index].reshape(-1, 1)
    x_bit = np.sign(solution_max_cut.reshape(2 * self.param['N'], 1))
    return x_bit.reshape(self.param['N'], 2, order='F')

```

#### 3.1.2 外层遗传算法超参数优化

保持振幅恒定的情形下，代码通过遗传算法（GA）搜索最优的超参数（如xi和weight）来间接优化相位。对于每组xi和weight，计算方向图性能。

其中每一组xi 和 weight，其优化的目标函数是根据score函数改编而来，

```python
def _evaluate(self, x, out, *args, **kwargs):
    xi_idx = x["xi"]
    weight_idx = x["weight"]
    xi = self.xi_values[x["xi"]]
    weight = self.weight_values[x["weight"]]

    left1, right1 = -55, -25
    left2, right2 = 25, 55

    param = {
        'theta_0': self.variables[0],
        'N': 32,
        'n_angle': 100,
        'encode_qubit': self.variables[1],
        'xi': xi,
        'dt': 0.3,
        'n_iter': 300,
        'weight': weight,
        'range_list': [[left1, right1], [left2, right2]],
        'range_list_weight': [1, 1],
    }
    encode_qubit = self.variables[1]

    phase_angle, amp = optimized_with_params(param)
    self.param_cache[(xi_idx, weight_idx)] = (phase_angle, amp)

    # 2. 归一化到 [0, 2π]
    phase_angle = np.angle(np.exp(1j * phase_angle)) + np.pi
    # 3. 离散化
    phase_angle = (
        np.round(phase_angle / (2 * np.pi) * (2 ** encode_qubit))
        / (2 ** encode_qubit) * (2 * np.pi)
    )

    amp_phase = []
    N = 32
    for i in range(N):
        amp_phase.append(amp[i] * np.exp(1.0j * phase_angle[i]))

    F = np.einsum('i, ij -> j', np.array(amp_phase), self.efield)
    FF = np.real(F.conj() * F)
    db_array = 10 * np.log10(FF / np.max(FF))

    x_theta = self.theta_array - self.theta_0
    mask = np.abs(x_theta) >= 30
    selected_values = db_array[mask] + 15
    a = max(np.max(selected_values) if selected_values.size > 0 else 0, 0)

    target = np.max(db_array)
    max_index = np.where(db_array == target)[0][0]

    theta_up = self.theta_array[-1]
    theta_down = self.theta_array[0]

    right_mask = db_array[max_index + 1:] <= -30
    if np.any(right_mask):
        theta_up = self.theta_array[max_index + 1 + np.argmax(right_mask)]

    left_mask = np.flip(db_array[:max_index]) <= -30
    if np.any(left_mask):
        theta_down = self.theta_array[max_index - 1 - np.argmax(left_mask)]

    W = theta_up - theta_down
    b = max(W - 6, 0)

    right_section = db_array[max_index + 1:-1]
    if right_section.size > 1:
        is_min_right = (right_section < np.roll(right_section, 1)) & (
            right_section < np.roll(right_section, -1)
        )
        if np.any(is_min_right):
            theta_min_up = self.theta_array[
                max_index + 1 + np.argmax(is_min_right)
            ]
        else:
            theta_min_up = self.theta_array[-1]
    else:
        theta_min_up = self.theta_array[-1]

    left_section = db_array[1:max_index][::-1]
    if left_section.size > 1:
        is_min_left = (left_section < np.roll(left_section, 1)) & (
            left_section < np.roll(left_section, -1)
        )
        if np.any(is_min_left):
            theta_min_down = self.theta_array[
                max_index - 1 - np.argmax(is_min_left)
            ]
        else:
            theta_min_down = self.theta_array[0]
    else:
        theta_min_down = self.theta_array[0]

    mask1 = np.abs(x_theta) >= 30
    mask2 = (x_theta >= (theta_min_up - self.theta_0)) | (
        x_theta <= (theta_min_down - self.theta_0)
    )
    combined_mask = mask1 & mask2
    selected_db = np.full_like(db_array, -np.inf)
    selected_db[combined_mask] = db_array[combined_mask] + 30
    c = np.max(selected_db)
    if not np.isfinite(c):
        c = 0.0

    direction_penalty = abs(self.theta_array[max_index] - self.theta_0)
    obj = 1000 - (100 * a + 80 * b + 20 * c)
    print(
        f"xi={xi:.4f}, weight={weight:.4f}，a: {a}, b: {b}, c: {c}, "
        f"max direction: {self.theta_array[max_index]}，obj: {obj}"
    )
    out["F"] = [-obj]

```

## 3.2 优化振幅，True的情形

代码通过联合优化振幅和相位，分三步实现波束成形设计：第一次相位优化  $\rightarrow$  第一次振幅优化  $\rightarrow$  联合优化振幅和相位。以下是具体实现逻辑：

<div align="center">
  <img src="https://cdn-mineru.openxlab.org.cn/result/2025-08-04/5396361b-55f4-4fbb-a3b6-a212ef42b5f5/b2ba7d08cc593cdc27af6309dc61cefd763f2cf59dd27e99efbd7ff2d6580e21.jpg" width="500"/>
  <p>图5 优化流程</p>
</div>

#### 3.2.1第一次相位优化：opt_phase方法

此处采用与样例一样的代码，通过梯度下降优化相位二进制解（x_bit），目标是最小化旁瓣能量与主瓣能量的比值：

$$
\dot{x}_i = y_i \tag{7}
$$

$$
\dot{y}_{i} = \left\{ \begin{array}{c}{-(\Delta -p(t))x_{i} + \xi \frac{\partial C(x_{i})}{\partial x_{i}},bSB}\\ {-(\Delta -p(t))sign(x_{i}) + \xi \frac{\partial C(sign(x_{i}))}{\partial x_{i}},dSB} \end{array} \right. \tag{8}
$$

$$
x_{i} = \dot{x}_{i}\Delta t,y_{i} = \dot{y}_{i}\Delta t,when|x_{i}|< 1 \tag{9}
$$

$$
x_{i} = sign(x_{i}),y_{i} = 0,when|x_{i}|\geq 1 \tag{10}
$$

代码中，  $\Delta = 0.5$  ，  $\mathfrak{p}(\mathfrak{t})$  为从0到1随时间均匀变化的系数。采用对不同范围加权后求平均值再除以主瓣信号强度的形式作为损失函数：

$$
C = \frac{wS}{|F(\theta_0)|^2} \tag{11}
$$

#### 3.2.2第一次振幅优化：opt_amp方法

对振幅进行第一次优化时，基于第一次相位优化的结果，采用与样例一样的代码，固定相位角phase_angle，通过Adam优化器最小化与相位优化相同的损失函数。

#### 3.2.3联合优化：joint_optimization方法

进行振幅相位联合优化时，基于第一次相位优化和第一次振幅优化的结果，将其作为初始种群，再同时优化振幅（实数变量）和相位二进制解（二元变量），通过混合变量遗传算法（MixedVariableGA）实现。

变量定义：

<table><tr><td colspan="2">代码片段3</td></tr><tr><td colspan="2">介绍：基于前面的结果，利用python对变量进行定义</td></tr><tr><td>01</td><td>vars = {</td></tr><tr><td>02</td><td>f&quot;amp_{i}&quot;::Real(bounds=(0,1)),          # 振幅（32个实数）</td></tr><tr><td>03</td><td>f&quot;x-bit_{i}{}&quot;]&quot;: Binary(),          # 相位二进制解（32×encode_qubit）</td></tr><tr><td>04</td><td>f&quot;phase_N&quot;: Integer(bounds=(2**encode_qubit,50)) # 相位离散化步数</td></tr><tr><td>05</td><td>}</td></tr></table>

目标函数：

改编Score函数，综合旁瓣抑制、主瓣宽度、主瓣内非主瓣区域能量等指标：

<table><tr><td>01</td><td>obj = 1000 - (100*a + 80*b + 20*c)</td></tr></table>

详细代码如下：
```python
  vars = {}
  for i in range(32):
      vars[f"amp_{i}"] = Real(bounds=(0, 1))          # 振幅（32 个实数）
      for j in range(encode_qubit):
          vars[f"x_bit_{i}_{j}"] = Binary()           # 相位二进制解（32 × encode_qubit）
  vars["phase_N"] = Integer(bounds=(2**encode_qubit, 50))  # 离散化步数
```

```python
def _evaluate(self, x, out, *args, **kwargs):
    amp = np.array([x[f"amp_{i}"] for i in range(32)])
    amp = amp / np.max(amp)

    x_bit = np.array(
        [[x[f"x_bit_{i}_{j}"] for j in range(encode_qubit)] for i in range(32)]
    )
    x_bit = np.round(x_bit).astype(int)

    phase_N = int(x["phase_N"])
    phase_angle = encode2(x_bit, self.theta_0, encode_qubit, phase_N)

    amp_phase = []
    for i in range(32):
        amp_phase.append(amp[i] * np.exp(1.0j * phase_angle[i]))

    F = np.einsum('i, ij -> j', np.array(amp_phase), self.efield)
    FF = np.real(F.conj() * F)
    db_array = 10 * np.log10(FF / np.max(FF))

    x_theta = self.theta_array - self.theta_0
    mask = np.abs(x_theta) >= 30
    selected_values = db_array[mask] + 15
    a = max(np.max(selected_values) if selected_values.size > 0 else 0, 0)

    target = np.max(db_array)
    max_index = np.where(db_array == target)[0][0]

    theta_up = self.theta_array[-1]
    theta_down = self.theta_array[0]

    right_mask = db_array[max_index + 1:] <= -30
    if np.any(right_mask):
        theta_up = self.theta_array[max_index + 1 + np.argmax(right_mask)]

    left_mask = np.flip(db_array[:max_index]) <= -30
    if np.any(left_mask):
        theta_down = self.theta_array[max_index - 1 - np.argmax(left_mask)]

    W = theta_up - theta_down
    b = max(W - 6, 0)

    right_section = db_array[max_index + 1:-1]
    if right_section.size > 1:
        is_min_right = (right_section < np.roll(right_section, 1)) & (
            right_section < np.roll(right_section, -1)
        )
        if np.any(is_min_right):
            theta_min_up = self.theta_array[
                max_index + 1 + np.argmax(is_min_right)
            ]
        else:
            theta_min_up = self.theta_array[-1]
    else:
        theta_min_up = self.theta_array[-1]

    left_section = db_array[1:max_index][::-1]
    if left_section.size > 1:
        is_min_left = (left_section < np.roll(left_section, 1)) & (
            left_section < np.roll(left_section, -1)
        )
        if np.any(is_min_left):
            theta_min_down = self.theta_array[
                max_index - 1 - np.argmax(is_min_left)
            ]
        else:
            theta_min_down = self.theta_array[0]
    else:
        theta_min_down = self.theta_array[0]

    mask1 = np.abs(x_theta) >= 30
    mask2 = (x_theta >= (theta_min_up - self.theta_0)) | (
        x_theta <= (theta_min_down - self.theta_0)
    )
    combined_mask = mask1 & mask2
    selected_db = np.full_like(db_array, -np.inf)
    selected_db[combined_mask] = db_array[combined_mask] + 30
    c = np.max(selected_db)
    if not np.isfinite(c):
        c = 0.0

    direction_penalty = abs(self.theta_array[max_index] - self.theta_0)
    obj = 1000 - (100 * a + 80 * b + 20 * c)
    constr = abs(self.theta_array[max_index] - self.theta_0) - 1
    out["F"] = [-obj]
    out["G"] = [constr]
def build_initial_solution(amp, x_bit, phase_N, encode_qubit, problem):
    solution = {}
    for i in range(32):
        solution[f"amp_{i}"] = amp[i]
    for i in range(32):
        for j in range(encode_qubit):
            solution[f"x_bit_{i}_{j}"] = int(x_bit[i][j])
    solution["phase_N"] = int(phase_N)
    return solution
```

#### 3.2.4 关键数据流总结

输入参数：theta_0（主瓣方向）、encode_qubit（编码比特数）、

流程：

第一次相位优化：通过梯度下降优化相位二进制解。

第一次振幅优化：固定相位，通过Adam优化器优化振幅。

联合优化：基于第一次相位优化和第一次振幅优化的结果，混合变量遗传算法同时优化振幅和相位。

输出结果：优化后的振幅amp和相位角phase_angle。

## 4 结果与分析

在其他条件相同的情况下，更改不同的抑制区间，或将S由一定范围内旁瓣信号强度的加权平均值，改为一定范围内信号强度的最大值，也会得到不同测试结果。提交测

试结果如表2所示。

目前最优测试结果返回的是419.8216。

表2测试结果  

<table><tr><td>方法</td><td>提交返回的score</td></tr><tr><td>QUBO_GA</td><td>390.0976</td></tr><tr><td>QUBO_GA</td><td>388.7642</td></tr><tr><td>QUBO_GA</td><td>419.8216</td></tr><tr><td>QUBO_GA</td><td>398.3819</td></tr></table>

在False情形下，调整参数，运行样例代码，可以得到相应的最优归一化信号强度 $|F(\theta)|^2$  示意图，此处以参数[75,3,False]为例：

<div align="center">
  <img src="https://cdn-mineru.openxlab.org.cn/result/2025-08-04/5396361b-55f4-4fbb-a3b6-a212ef42b5f5/8454dd3e72e30707173ae3c8469b5d852a46db9d0c4ea999a351e8c8359901e5.jpg" width="500"/>
  <p>图6 [75,3,False]参数下最优归一化信号强度 $|F(\theta)|^2$ 示意图</p>
</div>

同理，在True情形下，随着参数的调整，运行样例代码，也可以得到相应的最优归一化信号强度  $|F(\theta)|^2$  示意图，此处以参数[65,2,True]为例：

<div align="center">
  <img src="https://cdn-mineru.openxlab.org.cn/result/2025-08-04/5396361b-55f4-4fbb-a3b6-a212ef42b5f5/9770638e46233eb0408b2784576c16ebda8ba96ca6550f97c66e45516c66e6a6.jpg" width="500"/>
  <p>图7 [65,2,True] 参数下最优归一化信号强度 $|F(\theta)|^2$ 示意图</p>
</div>

结合上图，可以发现，成形方向得以压制，旁瓣强度低于主瓣方向的- 15dB，主瓣宽度减小，从而波束赋形场景得以优化。示意如下分析图：

<div align="center">
  <img src="https://cdn-mineru.openxlab.org.cn/result/2025-08-04/5396361b-55f4-4fbb-a3b6-a212ef42b5f5/29ca58afc07cbe28964b515a63b08dc58dad5601b609cad216bb7abd705454d6.jpg" width="500"/>
  <p>图8 运行结果分析示意图</p>
</div>

## 5 参考文献

[1] Jiang, Y., Ge, H., Wang, B. et al. Quantum- inspired Beamforming Optimization for Quantized Phase- only Massive MIMO Arrays. (2024). https://arxiv.org/abs/2409.19938[2] Zeng, Q., Cui, X., Liu, B. et al. Performance of quantum annealing inspired algorithms for combinatorial optimization problems. Commun. Phys. 7, 249 (2024). https://doi.org/10.1038/s42005- 024- 01705- 7

## 6 附录A

此处列出结果分析中未能展示的部分最优归一化信号强度  $|F(\theta)|^2$  示意图，其对应的参数包括[115, 2, True]等：

<!-- 图 9 -->
<div align="center">
  <img src="https://cdn-mineru.openxlab.org.cn/result/2025-08-04/5396361b-55f4-4fbb-a3b6-a212ef42b5f5/fbeb4455a4de7be62e54913c6039a2ca53e515b347ef436e41a9b438072634a9.jpg" width="500"/>
  <p>图 9 其他结果图（1）</p>
</div>

<!-- 图 10 -->
<div align="center">
  <img src="https://cdn-mineru.openxlab.org.cn/result/2025-08-04/5396361b-55f4-4fbb-a3b6-a212ef42b5f5/c2a15adc4b91a9301e00b577601743ef5436983f6bcc4b250e533f7360295679.jpg" width="500"/>
  <p>图 10 其他结果图（2）</p>
</div>

<!-- 图 11 -->
<div align="center">
  <img src="https://cdn-mineru.openxlab.org.cn/result/2025-08-04/5396361b-55f4-4fbb-a3b6-a212ef42b5f5/3907fb92c962bb4484588a19584b7bc3670d2e61f4b25949289d9d78f192a720.jpg" width="500"/>
  <p>图 11 其他结果图（3）</p>
</div>

<!-- 图 12 -->
<div align="center">
  <img src="https://cdn-mineru.openxlab.org.cn/result/2025-08-04/5396361b-55f4-4fbb-a3b6-a212ef42b5f5/eea9da2082ae1ec3e2ba6f09df16f77a74c803c9866f8a37972d97bfc4fe6133.jpg" width="500"/>
  <p>图 12 其他结果图（4）</p>
</div>


