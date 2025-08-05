import numpy as np
from itertools import product
from qutip import basis, tensor, Qobj, fidelity

# 假设 MindQuantum 中提供了以下模块和门
from mindquantum import Circuit, Simulator
from mindquantum.core.gates import X, RZ, RotPauliString,Measure,RX,CNOT#, MeasureAll, Reset  # 注意：部分门（如 Reset、MeasureAll）可能需要根据实际版本调整
# 假设 MindQuantum 支持噪声模型
from mindquantum.core.gates import DepolarizingChannel
from mindquantum.core.operators import Hamiltonian, QubitOperator
 
#==============================================================================
#=========================== Define three ansatz  =============================
#==============================================================================
 

class ansatz:
    def __init__(self, N, T, Nt):
        '''
        输入：量子比特数 N；总时间 T；Trotter 步数 Nt
        '''
        # 默认常数
        self.C0, self.J0 = 1, 1  
        self.N = N
        self.Nt = Nt
        self.di, self.df = 0, N - 1  # 控制函数 d(t) 的边界条件
        self.tlist = np.linspace(0, T, self.Nt)  # 离散时间列表
        self.dt = self.tlist[1] - self.tlist[0]    # 单步时长
        self.Clist0 = np.ones(self.Nt) * self.C0    # 常数 C0 列表
        self.Dlist0 = np.linspace(self.di, self.df, self.Nt)  # 线性变化的 d(t)
        self.psi_in, self.psi_tar = self.get_boundary_state()  # 初始态与目标态
        self.psi_fin = self.psi_in

    def Bn(self, C, d, N, xth):
        '''
        定义 Bn(t,x) 函数
        '''
        return -0.5 * C * (xth - d) ** 2

    def get_boundary_state(self):
        '''
        返回控制问题中初始态和目标态（利用 qutip 构造）
        '''
        spin_up = basis(2, 1)
        spin_down = basis(2, 0)
        state = spin_up  # 待传输的自旋状态（也可考虑线性组合）
        psi_inlist, psi_tarlist = [], []
        for i in range(self.N):
            if i == self.N - 1:
                psi_inlist.append(state)
            else:
                psi_inlist.append(spin_down)
            if i == 0:
                psi_tarlist.append(state)
            else:
                psi_tarlist.append(spin_down)
        psi_in = tensor(psi_inlist).full()
        psi_tar = tensor(psi_tarlist).full()
        return psi_in, psi_tar
    

    def correlated_I(self, xlist):
        '''
        使用 MindQuantum 构建电路（无可调参数控制函数 d(t)）
        输入：xlist 为离散时间点上 d(t) 的取值
        '''
        # 构建电路（指定 N 个量子比特）
        qc = Circuit()
        # 初始状态准备：对 0 号比特施加 X 门（将 |0> 翻转为 |1>），模拟原代码的初始化
        qc += X.on(0)
        th = 0
        # 对于每个时间步，依次添加演化门
        for t in self.tlist:
            C = self.Clist0[th]
            d = xlist[th]
            theta1 = -0.5 * self.J0 * self.dt
            # H1 部分：对相邻比特施加 Rxx 和 Ryy 门（利用 RotPauliString 实现）
            for k in range(self.N - 1):
                qc += RotPauliString('XX', 2 * theta1).on([k, k+1])
                qc += RotPauliString('YY', 2 * theta1).on([k, k+1])
            # H2 部分：对每个比特施加 Rz 门
            for j in range(self.N):
                theta2 = self.dt * self.Bn(C, d, self.N, j)
                qc += RZ(2 * theta2).on(j)
            th += 1
        # 利用状态向量模拟器进行无噪声模拟
        sim = Simulator('mqvector', self.N)
        psi1 = qc.get_qs()  # 返回最终态向量，形状为 (2**N,)
        
        # 计算与目标态的保真度（qutip 用法）
        fid = fidelity(Qobj([psi1]), Qobj(self.psi_tar))
        loss = 1 - fid
        self.qc=qc
        return loss, psi1  

    def correlated_II(self, xlist):
        '''
        使用 MindQuantum 构建电路（控制函数包括 d(t) 和 C(t)）
        输入：xlist 为一维数组，前 Nt 个元素对应 d(t)，后 Nt 个元素对应 C(t)
        '''
        N = self.N
        dlist = xlist[:self.Nt]
        Clist = xlist[self.Nt:]
        # 边界条件：d(0)=di, d(T)=df
        dlist[0] = self.di
        dlist[-1] = self.df

        qc = Circuit()
        qc += X.on(0)
        th = 0
        for t in self.tlist:
            C = Clist[th]
            d = dlist[th]
            theta1 = -0.5 * self.J0 * self.dt
            for k in range(N - 1):
                qc += RotPauliString('XX', 2 * theta1).on([k, k+1])
                qc += RotPauliString('YY', 2 * theta1).on([k, k+1])
            for j in range(N):
                theta2 = self.dt * self.Bn(C, d, N, j)
                qc += RZ(2 * theta2).on(j)
            th += 1
        sim = Simulator('mqvector', N)
        psi1 = qc.get_qs()
   
        fid = fidelity(Qobj([psi1]), Qobj(self.psi_tar)) 
        loss = 1 - float(fid)
        self.qc=qc
        return float(loss), psi1 

    def uncorrelated(self, xlist):
        '''
        使用 MindQuantum 构建电路（仅对训练层参数进行优化）
        输入：xlist 为长度为 Nt*N 的一维数组，重新 reshape 成 (Nt, N)
        '''
        qc = Circuit()
        qc += X.on(0)
        th = 0
        thetalist = np.reshape(xlist, newshape=(self.Nt, self.N))
        for t in self.tlist:
            theta1 = -0.5 * self.J0 * self.dt
            for k in range(self.N - 1):
                qc += RotPauliString('XX', 2 * theta1).on([k, k+1])
                qc += RotPauliString('YY', 2 * theta1).on([k, k+1])
            for j in range(self.N):
                theta2 = thetalist[th][j]* self.dt
                qc += RZ(2 * theta2).on(j)
            th += 1
        # 无噪声模拟
        sim = Simulator('mqvector', self.N)
        psi1 = qc.get_qs()
  
        fid = fidelity(Qobj([psi1]), Qobj(self.psi_tar)) 
        loss = 1 - float(fid)

        self.qc=qc
        return float(loss), psi1
    
    def hardware_efficient(self, params):
        """
        构建简化的N-qubit Hardware-Efficient Ansatz:
            - 单比特旋转: 每比特每层交替 Rz 和 Rx
            - 纠缠: 线性最近邻 CNOT (0→1→2→...)
            - 参数总数: 2 * n_qubits * layers
        
        参数:
            n_qubits (int): 量子比特数
            layers (int): 层数 (默认: Nt)
        
        返回:
            QuantumCircuit: 参数化的HEA电路,loss, 末态 psi1
        """
        qc = Circuit()
        # qc += X.on(0)
        # for t in self.tlist:
        #     theta1 = -0.5 * self.J0 * self.dt
        #     for k in range(self.N - 1):
        #         qc += RotPauliString('XX', 2 * theta1).on([k, k+1])
        #         qc += RotPauliString('YY', 2 * theta1).on([k, k+1])
        n_qubits = self.N
        layers = self.Nt
        params = np.reshape(params, (layers, n_qubits,2))
        for _ in range(layers):
            # 1. 单比特旋转层 (Rz + Rx)
            param_idx = 0
            for qubit in range(n_qubits):
                qc+=RZ(params[_,param_idx,0]).on(qubit)
                qc+=RX(params[_,param_idx,1]).on(qubit)
                param_idx += 1
            # 2. 线性CNOT纠缠层 (0→1→2→...)
            for src in range(n_qubits - 1):
                qc+=CNOT.on(src, src + 1)
        # 无噪声模拟
        psi1 = qc.get_qs()
        fid = fidelity(Qobj([psi1]), Qobj(self.psi_tar)) 
        loss = 1 - float(fid)
        self.qc=qc
        return float(loss), psi1
    

    def uncorrelated_noise(self, xlist,p_tot):
 
        rdw = np.random.random(1)[0]
        p_reset = p_tot*rdw
        p_meas = p_tot*(1-rdw)

        sim = Simulator('mqvector', 1)
        qc = Circuit()
        for i in range(self.N):
            qc += DepolarizingChannel(p_reset).on(i)

        qc += X.on(0)
        th = 0
        thetalist = np.reshape(xlist, newshape=(self.Nt, self.N))
        for t in self.tlist:
            theta1 = -0.5 * self.J0 * self.dt
            for k in range(self.N - 1):
                qc += RotPauliString('XX', 2 * theta1).on([k, k+1])
                qc += RotPauliString('YY', 2 * theta1).on([k, k+1])
            for j in range(self.N):
                theta2 = thetalist[th][j]* self.dt
                qc += RZ(2 * theta2).on(j)
            th += 1

        for i in range(self.N):
            qc += DepolarizingChannel(p_meas).on(i)

        for i in range(self.N):
            qc += Measure().on(i)

        

        # 噪声模拟
        sim.reset() 
        sim = Simulator('mqvector', self.N)
        sim.apply_circuit(qc)
        #shot_num= 4096
        #counts_noise = sim.sampling(qc, {'a': 1.1, 'b': 2.2}, shots= shot_num, seed=42).data
        #print('counts_noise',counts_noise)
        ham = Hamiltonian(QubitOperator(f'Z{0}'))
        fid = sim.get_expectation(ham).real 
        #psi1=sim.get_qs()
       
        #fid = fidelity(Qobj([psi1]), Qobj(self.psi_tar))
        # mea_basi0 = ['0', '1']
        # mea_basi0 = list(product(mea_basi0, repeat=self.N))
        # mea_basi = []
        # for mea in mea_basi0:
        #     str0 = mea[0]
        #     for i in range(len(mea) - 1):
        #         str0 += mea[i + 1]
        #     mea_basi.append(str0)
        # stlist = [mea_basi[2**(self.N-i-1)] for i in range(self.N)] #['100000', '010000', '001000', '000100', '000010', '000001']
        # #print('stlist', stlist)
        # st0  = stlist[0] #'10000...
        
        # if st0 in counts_noise.keys():
        #     fid = counts_noise[st0] / shot_num
        # else:
        #     fid = 0

        # p=0
        # for st in stlist:
        #     if st in counts_noise.keys():
        #         px =  counts_noise[st]/shot_num
        #         #print('st',st,'px',px)
        #     else:
        #         px = 0
        #     p+=px
        print('fid',fid)#fid = counts_noise[st0] / shot_num
        loss = 1 - fid
        self.qc=qc
        return loss, qc 
