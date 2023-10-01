# 基于昇思 MindSpore Quantum，实现用于求解分子体系本征态的非变分量子算法  

本项目主要关注 PSHO 算法的间接法实现方案。下面对该方案的代码进行展示。该代码可在华为云 CouldIDE，或本地 Linux 系统下执行。

导入所需依赖项

```python
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum import *
from mindquantum.algorithm.nisq import generate_uccsd
```

定义分子结构，以 H2 为例

```python
blen = 0.74 # 键长
geom = [["H", [0.0, 0.0, 0.0 * blen]],
        ["H", [0.0, 0.0, 1.0 * blen]],]
basis = "sto3g" # 基底
spin = 0
```

获取分子数据

```python
molecule_data = MolecularData(geom, basis, multiplicity=2 * spin + 1)
molecule_of = run_pyscf(molecule_data, run_scf=0, run_ccsd=1, run_fci=1) 
fci_res = molecule_of.fci_energy # fci 计算结果
print('fci 结果为：', fci_res) 
```

```
fci 结果为： -1.1372838344885023
```

构造所需哈密顿量

```python
_,_,_,ham_mol, mol_n_qubits, n_electrons = generate_uccsd(molecule_data, threshold=0)
# ham_mol 是一个 QubitOperator
# 形如： -0.0971 [] + 0.1714 [Z0] + 0.1714 [Z1] + ...

ham_list0 = [] # 用于承接分子哈密顿算符的每一个子项（除常数项外）。
# 形如： [0.1714*delta [Z0], 0.1714*delta [Z1], -0.2234*delta [Z2], ...]

for i, term in enumerate(ham_mol.terms): # 去掉第一项常数项，该项不用于构造线路。
    if i ==0:
        constant =  ham_mol.get_coeff(term).const.real
        print('常数项系数为：', constant)
        constant_term = term
    else:
        ham_list0.append(QubitOperator(term, ham_mol.get_coeff(term).const) *  ParameterResolver('delta') )  # 给每一项都加上同一个系数。后面用于构造含参时间演化算符线路。

ham_list1 = [] # 操作 ham_list0, 后面作为对时间演化算符进行 trotter 分解时依次执行的字符串
# 形如：[0.1714*t [Z0], 0.1714*t [Z1], ..., 0.1206*t [Z1 Z3], 0.1744*t [Z2 Z3], 0.1206*t [Z1 Z3], ...,    0.1714*t [Z1], 0.1714*t [Z0] ]

for term in ham_list0: # trotter 近似的前半部分
    ham_list1.append(term)

for i in range(len(ham_list0)-2, -1, -1): # trotter 近似的后半部分
    ham_list1.append(ham_list0[i])

ham_H = Hamiltonian(ham_mol - QubitOperator(constant_term, constant) ) 
# 论文中 H 的期望值（去掉了常数项）
# 形如：  0.1714 [Z0] + 0.1714 [Z1] + ...

ham_HZ = Hamiltonian((ham_mol - QubitOperator(constant_term, constant)) * QubitOperator(f'Z{mol_n_qubits}', -1)) 
# 论文中 -ZH 的期望值 （去掉了常数项）
# 形如： -0.1714 [Z0 Z4] + -0.1714 [Z1 Z4] + ...

ham_IZ = Hamiltonian(QubitOperator(f'Z{mol_n_qubits}', -1))                
# 论文中 -ZI 的期望值
# 形如： -1 [Z4] 
```

定义一个可以求组合数的函数

```python
def com(n,m): # 用于求组合数 C_nm，n >= m
    res0 = 1
    for i in range(n-m+1, n+1, 1):
        res0 *= i
    res1 = 1
    for i in range(1,m+1):
        res1 *= i
    return res0/res1
```

构造量子线路

```python
tau = 1.38 # 演化时间 tau = (np.pi/2) / E_0
delta = 0.01 # Trotter 近似的
n = 10 # 幂数

hatree = Circuit([X.on(i) for i in range(n_electrons)])   # 生成 ref 态
hatree += I.on(mol_n_qubits-1) # 加上一个单位门，用于扩展量子线路的宽度，使其与时间演化分子的比特数对齐，从而方便在后面求 ref 态的期望值（分子）

### 求解分子和分母的第一项：
An1 = 0
Bn1 = 0

for k in range(n):
    print(k)
    t = tau*(n-k)
    num_slices = int(t/delta) # trotter 分解级数

    # 算符 e^{i*H*t} 
    circ1 = Circuit()
    for i, term in enumerate(ham_list1):
        if i==int(len(ham_list1)/2):
            circ1 += TimeEvolution(term.subs({'delta':-delta})).circuit
        else:
            circ1 += TimeEvolution(term.subs({'delta':-delta/2})).circuit
    circ1 *= num_slices

    # 算符 e^{-i*H*t}
    circ2 = Circuit()
    for i, term in enumerate(ham_list1):
        if i==int(len(ham_list1)/2):
            circ2 += TimeEvolution(term.subs({'delta':delta})).circuit
        else:
            circ2 += TimeEvolution(term.subs({'delta':delta/2})).circuit
    circ2 *= num_slices

    circ = Circuit()
    circ += hatree

    circ += H.on(circ1.n_qubits) # 辅助比特在最下面
    circ += S.on(circ1.n_qubits)
    circ += H.on(circ1.n_qubits)

    circ += controlled(circ1)(circ1.n_qubits) # 时间演化算符线路受辅助比特控制
    circ += X.on(circ1.n_qubits)
    circ += controlled(circ2)(circ1.n_qubits)
    circ += X.on(circ1.n_qubits)

    circ += H.on(circ1.n_qubits)
    circ += S.on(circ1.n_qubits)
    circ += H.on(circ1.n_qubits)
    
    sim1 = Simulator('mqvector', circ.n_qubits)
    sim1.apply_circuit(circ)

    value1 = sim1.get_expectation(ham_HZ).real # 论文中 -ZH 的期望值
    value2 = sim1.get_expectation(ham_IZ).real   # 论文中 -ZI 的期望值

    An1 += (-1)**k*2*com(2*n, k)*value1
    Bn1 += (-1)**k*2*com(2*n, k)*value2

### 求分子和分母的第二项

sim2 = Simulator('mqvector', hatree.n_qubits)
sim2.apply_circuit(hatree)

An2 = com(2*n, n)*(-1)**n*(sim2.get_expectation(ham_H).real)
Bn2 = com(2*n, n)*(-1)**n

res =  (An1+An2)/(Bn1+Bn2)

print('计算结果为：', res + constant)

print('与 fci 的差值为：', abs(fci_res - (res + constant)))

```

