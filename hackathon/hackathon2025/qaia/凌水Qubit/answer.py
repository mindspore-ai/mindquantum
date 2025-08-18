import numpy as np
import time


def optimized(variables):
    def bit2int(A, group_len, weights):
        # 函数作用：将一串二进制比特转化为十进制
        num_rows, num_bits = A.shape
        num_groups = num_bits // group_len
        A_reshaped = A.reshape(num_rows, num_groups, group_len)

        # 使用点积，乘以权重矩阵即可
        A_bit = np.tensordot(A_reshaped, weights, axes=([2], [0]))
        return A_bit

    def make(N1, Q1, phase_bits1, amplitude_bits1, num_of_array1, num_of_Qbit1):
        # 函数作用：观察量子比特种群，由叠加态坍缩为确定态
        c = np.random.rand(N1, num_of_Qbit1)
        P1 = np.array(c < Q1 ** 2, dtype='int')  # 生成随机数与 Q^2 进行比较，模拟量子坍缩过程
        if amplitude_bits1 > 0:  # 优化振幅
            # 权重数组
            phase_weight = 2 ** np.arange(phase_bits1 - 1, -1, -1)
            amplitude_weight = 2 ** np.arange(amplitude_bits1 - 1, -1, -1)

            # 将二进制转为十进制，计算 相位P_alpha 和 振幅P_beta
            P_alpha1 = bit2int(P1[:, 0:num_of_array1 * phase_bits1], phase_bits1, phase_weight) * (
                    np.pi / (2 ** (phase_bits1 - 1)))
            P_beta1 = (2 * bit2int(P1[:, num_of_array1 * phase_bits1: num_of_Qbit1], amplitude_bits1,
                                   amplitude_weight) + 1) * (1 / (2 ** (amplitude_bits1 + 1)))
        else:  # 不优化振幅
            # 权重数组
            phase_weight = 2 ** np.arange(phase_bits1 - 1, -1, -1)
            P_alpha1 = bit2int(P1[:, 0:num_of_array1 * phase_bits1], phase_bits1, phase_weight) * (
                    np.pi / (2 ** (phase_bits1 - 1)))

            # 不优化振幅时振幅全部初始化为1
            P_beta1 = np.ones((N1, num_of_array1))
        return P1, P_alpha1, P_beta1

    def fitness(N1, num_of_array1, theta_01, P_alpha1, P_beta1):
        # 函数作用：求适应度值
        def islocalmin_rows_with_flats(beam1):
            # 函数作用：寻找局部最小值的位置
            padded = np.pad(beam1, ((0, 0), (2, 2)), mode='edge')

            l2 = padded[:, :-4]
            l1 = padded[:, 1:-3]
            center = padded[:, 2:-2]
            r1 = padded[:, 3:-1]
            r2 = padded[:, 4:]

            # 严格极小值
            is_min = (center < l1) & (center < r1)

            # 平台极小值：中心值等于邻居，且平台整体低于外部
            is_flat = (center == l1) & (center == r1) & (center < l2) & (center < r2)

            TF1 = is_min | is_flat
            return TF1

        def find_theta_1pand_2p(beam1, max_index1, p_or_not):
            # 函数作用：寻找θ1、θ2或θ1‘、θ2’
            if p_or_not:  # 寻找θ1‘、θ2’
                TF = np.where(beam1 > -30, 0, 1)
                TF = TF.astype(bool)
            else:  # 寻找θ1、θ2
                TF = islocalmin_rows_with_flats(beam1)
            M1, N1 = beam1.shape

            # 位置矩阵（和 MATLAB 中 repmat 一样）
            col_indices = np.tile(np.arange(N1), (M1, 1))  # shape: (M, N)

            # 保留 TF 为 True 的位置，其他设为 NaN
            min_idx = np.where(TF, col_indices, np.nan)  # shape: (M, N)

            # 将 max_index 转换为列向量广播
            max_index_col = max_index1.reshape(-1, 1)

            # 距离矩阵（负值表示在 max_index 左边，正值在右边）
            distances = min_idx - max_index_col  # shape: (M, N)

            # 负方向距离：大于等于0置为 -inf
            dis_neg = np.where(distances < 0, distances, -np.inf)
            theta_1p_index1 = np.nanargmax(dis_neg, axis=1)
            theta_1p_index1 = np.where(theta_1p_index1 == 0, 0, theta_1p_index1)
            theta_1p_index1 = np.where(theta_1p_index1 > max_index1, 0, theta_1p_index1)

            # 正方向距离：小于等于0置为 inf
            dis_pos = np.where(distances > 0, distances, np.inf)
            theta_2p_index1 = np.nanargmin(dis_pos, axis=1)
            theta_2p_index1 = np.where(theta_2p_index1 == 0, N1 - 1, theta_2p_index1)

            # 防呆处理，替代非法值
            theta_2p_index1 = np.where(theta_2p_index1 < max_index1, N1 - 1, theta_2p_index1)
            return theta_1p_index1, theta_2p_index1

        # 为加速运算，使用矩阵运算代替for循环
        step = 0.005
        Theta = np.arange(0, np.pi, step)  # 角度扫描，精度0.005（弧度制）
        # 求E(θ)
        E_dB = -np.minimum(12 * ((Theta / np.pi * 180 - 90) / 90) ** 2, 30)
        E = 10 ** (E_dB / 10)
        # 求An(θ)
        array = np.arange(1, num_of_array1 + 1)
        # array是长为num_of_array的行向量
        I = P_beta1 * np.exp(1j * P_alpha1)  # I是N * num_of_array的矩阵
        # P_beta1和P_alpha1都是N * num_of_array的矩阵
        I = np.tile(I[:, :, np.newaxis], [1, 1, len(Theta)])  # I是N * num_of_array * step的矩阵
        exp1 = np.exp(np.pi * 1j * np.outer(array, np.cos(Theta)))  # exp1是num_of_array * step的矩阵
        exp1 = np.tile(exp1[np.newaxis, :, :], [N1, 1, 1])  # exp1是N1 * num_of_array * step的矩阵
        A = I * exp1  # A是N1 * num_of_array * step的矩阵

        # 求F(θ)，求和之后，A变成N1 * step的矩阵，因此E要扩展一个维度
        F_theta = np.abs(np.tile(E[np.newaxis, :], (N1, 1)) * (np.sum(A, axis=1) * np.sum(np.conj(A), axis=1)))
        beam = 10 * np.log10(F_theta / np.tile(np.max(F_theta, axis=1)[:, np.newaxis], [1, len(Theta)]))
        # 求最大值索引
        max_index = np.argmax(beam, axis=1)  # 求第二个维度上的最大值索引，结果是一维N1的数组，每个元素对应一个theta

        # 找 θ1 和 θ2
        theta_1_index, theta_2_index = find_theta_1pand_2p(beam, max_index, False)
        theta_1p_index, theta_2p_index = find_theta_1pand_2p(beam, max_index, True)

        # 找 θ1' 和 θ2'
        theta_1p = (theta_1p_index * step / np.pi) * 180
        theta_2p = (theta_2p_index * step / np.pi) * 180
        # 定义 30° 区间
        theta_30_up_index = int(np.ceil((((theta_01 + 30) / 180) * np.pi) / step))
        theta_30_down_index = int(np.floor((((theta_01 - 30) / 180) * np.pi) / step))

        # 目标函数项
        # 1. 30°外小于 -15dB
        a = np.zeros([N1])
        mask_a = np.hstack([np.arange(0, theta_30_down_index), np.arange(theta_30_up_index, beam.shape[1])])
        for ai in range(N1):
            a[ai] = np.max(beam[ai, mask_a])
        a[a < -15] = -15
        # 2. 主瓣宽度小于6°

        b = theta_2p - theta_1p
        b[b < 6] = 6

        # 3. 正负30°内小于 -30dB
        c = np.zeros([N1])
        for ci in range(N1):
            mask_c = np.hstack(
                [np.arange(theta_30_down_index, theta_1_index[ci], dtype='int'),
                 np.arange(theta_2_index[ci], theta_30_up_index, dtype='int')])
            if len(mask_c) == 0:
                c[ci] = np.inf
            else:
                c[ci] = np.max(beam[ci, mask_c])
        c[c < -30] = -30

        # 4. 指向角偏差
        d = np.abs(max_index * step * 180 / np.pi - theta_01)

        # 偏差超过1°的设为inf
        f = -(1000 - 100 * (a + 15) - 80 * (b - 6) - 20 * (c + 30))
        f[d > 0.7] = np.inf  # 因为精度问题，d > 0.7而不是d > 1
        para = np.stack((a, b, c, d), axis=1)
        return beam, f, para

    def update(N1, num_of_Qbit1, Q1, P1, B1, P_f_update, B_f_update):  # 核心函数，用于更新量子比特种群
        # 函数作用：用旋转门更新量子比特
        theta3 = 0.01 * np.pi  # 量子旋转门步长，即每次更新时对量子比特的旋转角度
        theta5 = -0.01 * np.pi
        threshold = 0.44 * np.pi  # 阈值，保证量子比特振幅不会太大或者太小
        P_f_update = np.nan_to_num(P_f_update, nan=0.0)  # 保证不会出现nan值
        B_f_update = np.nan_to_num(B_f_update, nan=0.0)
        state = P_f_update - B_f_update
        delta_theta = np.zeros([N1, num_of_Qbit1])
        delta_theta[(P1 == 0) & (B1 == 1) & (np.tile(state[:, np.newaxis], [1, num_of_Qbit1]) > 0)] = theta3
        # 当P较为优秀且B为1，P为0时，将对应的量子比特向0旋转一个步长
        delta_theta[(P1 == 1) & (B1 == 0) & (np.tile(state[:, np.newaxis], [1, num_of_Qbit1]) > 0)] = theta5
        # 当P较为优秀且P为1，B为0时，将对应的量子比特向1旋转一个步长
        Q1 = np.sin(delta_theta) * np.sqrt(1 - Q1 ** 2) + np.cos(delta_theta) * Q1
        Q1[Q1 > np.sin(threshold)] = np.sin(threshold)
        Q1[Q1 < np.cos(threshold)] = np.cos(threshold)
        # 设定阈值，大于或者小于阈值都会被重新赋值
        return Q1

    def Select_B(B1, B_f1, B_alpha1, B_beta1, B_beam1, B_para1, Q1):
        # 函数作用：精英选择
        N1 = np.size(B_f1[:])
        C, Front_index = np.unique(B_f1, return_index=True)
        # 将分数从小到大排序
        if len(C) < N1 / 2:
            Front_index = np.argsort(B_f1)
        Next = Front_index[0:int(np.ceil(N1 / 2))]
        # 得到排序后前N1 / 2个个体的序号
        B1 = B1[Next, :]  # 迁移对应序号的B、B_f、B_alpha、B_beta、B_beam、B_paraQ
        B_f1 = B_f1[Next]

        B_alpha1 = B_alpha1[Next, :]
        B_beta1 = B_beta1[Next, :]
        B_beam1 = B_beam1[Next, :]
        B_para1 = B_para1[Next, :]
        Q1 = Q1[Next, :]
        return B1, B_f1, B_alpha1, B_beta1, B_beam1, B_para1, Q1

    # 主程序
    start_time = time.time()  # 计时开始
    theta_0 = variables[0]  # 读入优化中心角度变量
    phase_bits = variables[1]  # 读入相位比特数目
    if_amplitude = variables[2]  # 读入是否优化振幅
    num_of_array = 32  # 天线阵子数目
    if if_amplitude:
        amplitude_bits = 6  # 振幅优化时，振幅量化比特数目
    else:
        amplitude_bits = 0  # 不优化则为零
    array_bits = phase_bits + amplitude_bits  # 每个天线阵子对应的量子比特数目
    num_of_Qbit = num_of_array * array_bits  # 每个种群包含的量子比特数目
    N = 80  # 种群个体数量
    i = 1  # 指代当前迭代次数

    # 初始化量子比特种群中的每一个量子比特振幅为1/sqrt(2)
    Q = (1 / np.sqrt(2)) * np.ones([N, num_of_Qbit])

    # 观测量子比特种群生成第一代新生种群，同时也是精英种群
    B, B_alpha, B_beta = make(N, Q, phase_bits, amplitude_bits, num_of_array, num_of_Qbit)

    # 计算第一代新生种群的得分情况
    B_beam, B_f, B_para = fitness(N, num_of_array, theta_0, B_alpha, B_beta)
    while 1:
        i += 1  # 迭代次数加一
        # 观测量子比特种群产生新生种群
        P, P_alpha, P_beta = make(N, Q, phase_bits, amplitude_bits, num_of_array, num_of_Qbit)

        # 计算新生种群的适应度分数
        P_beam, P_f, P_para = fitness(N, num_of_array, theta_0, P_alpha, P_beta)

        # 更新Q-bit种群
        Q = update(N, num_of_Qbit, Q, P, B, P_f, B_f)

        # 根据上一代精英种群和本代新生种群的得分选出新的精英种群
        B, B_f, B_alpha, B_beta, B_beam, B_para, Q = Select_B(
            np.vstack([B, P]), np.concatenate([B_f, P_f]), np.concatenate([B_alpha, P_alpha]),
            np.concatenate([B_beta, P_beta]), np.concatenate([B_beam, P_beam]), np.concatenate([B_para, P_para]),
            np.vstack([Q, Q])
        )
        if time.time() - start_time > 89:  # 超时算法终止
            break
    return B_alpha[0, :], B_beta[0, :]  # 返回最优个体的相位和振幅
