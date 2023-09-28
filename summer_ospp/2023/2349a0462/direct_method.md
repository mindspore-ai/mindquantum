# 基于昇思 MindSpore Quantum，实现用于求解分子体系本征态的非变分量子算法  

本文件主要关注 PSHO 算法的直接法实现方案。下面对该方案的代码进行展示。该代码可在华为云 CouldIDE，或本地 Linux 系统下执行。

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

ham_H = Hamiltonian(ham_mol - QubitOperator(constant_term, constant) ) # 论文中 H 的期望值（去掉了常数项）
# 形如：  0.1714 [Z0] + 0.1714 [Z1] + ...
```

构造量子线路

```python
tau = 1.38 # 演化时间 tau = (np.pi/2) / E_0
delta = 0.01
num_slices = int(tau/delta)

# 算符 e^{i*H*tau} 
circ1 = Circuit()
for i, term in enumerate(ham_list1):
    if i==int(len(ham_list1)/2):
        circ1 += TimeEvolution(term.subs({'delta':-delta})).circuit
    else:
        circ1 += TimeEvolution(term.subs({'delta':-delta/2})).circuit
circ1 *= num_slices

# 算符 e^{-i*H*tau}
circ2 = Circuit()
for i, term in enumerate(ham_list1):
    if i==int(len(ham_list1)/2):
        circ2 += TimeEvolution(term.subs({'delta':delta})).circuit
    else:
        circ2 += TimeEvolution(term.subs({'delta':delta/2})).circuit
circ2 *= num_slices

hatree = Circuit([X.on(i) for i in range(n_electrons)])   # 生成 ref 态态的期望值（分子）

def sigma():
    circ = Circuit()

    circ += H.on(4) # 辅助比特在最下面
    circ += S.on(4)
    circ += H.on(4)

    circ += controlled(circ1)(4) # 时间演化算符线路受辅助比特控制
    circ += X.on(4)
    circ += controlled(circ2)(4)
    circ += X.on(4)

    circ += H.on(4)
    circ += S.on(4)
    circ += H.on(4)
    circ += Measure().on(4)
    return circ


sim = Simulator('mqvector', 5)
sim.apply_circuit(hatree)
state = sim.get_qs()

error = []
j = 0 # 辅助比特连续测试为 0 的次数
for i in range(10):
    sim.apply_circuit(sigma())
    state_str = sim.get_qs(True)
    state_tem = sim.get_qs()
    if state_str[-6] == '0': # 如果辅助比特测量结果为 0 就继续运行下一轮，否则无视上一轮线路的操作，重新来
        j += 1
        state = state_tem
        res = sim.get_expectation(ham_H)
        print(j, abs(fci_res - (res + constant)))
        error.append(abs(fci_res - (res + constant)))
    else:
        sim.set_qs(state)

print('运行完啦!')
print(error)

```

