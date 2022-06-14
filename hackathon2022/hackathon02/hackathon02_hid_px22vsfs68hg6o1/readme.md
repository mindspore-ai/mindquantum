# 黑客松赛题-量子化学模拟:基态能求解的浅层线路设计

Variational Quantum Eigensolver（VQE）算法是一种用于计算分子基态能量的量子化学模拟方法，它可以结合经典计算机，利用当前含噪的量子计算机解决一些化学问题。现有常用的线路设计模板通常难以平衡计算效率和计算精度的问题，而线路设计方案是这种算法领域中被研究的一个主要问题。

## 赛题要求：

- 我们对于不同的测试分子案例，选择基矢为sto-3g。根据给出的分子体系设计量子线路并试进行各种优化，使计算的基态能量尽可能达到化学精度(0.0016Ha)。
- 要求设计线路需要有可扩展性，能根据不同的化学分子系统给出相应的量子线路，测试案例包括两个给出的分子；举办方会再评测一个分子。
- 程序要求基于mindquantum，可基于现有的ansatz进行修改，也可以自行创作新的ansatz。
- 参数优化过程可以基于梯度的方法或非梯度的方法。
- 计算结果利用模板中的`main.py`中的Plot作图。


## 测试数据：

- 分子1（40%）：LiH，比特数：12，键长点：[0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
- 分子2（30%）：CH4，比特数：18，键长点：[0.4, 0.8]
- 分子3（30%）：举办方测试，不对外公布

##项目使用的主要方法：

QUCC量子线路
1.先通过分子轨道比特编码:分为single和double进行激发，使用Jordan-Wigner编码。
再将参数送入量子线路中。
主要修改部分：
-（1）选用分子哈密顿量的其中一部分作用于Ansatz:使用FermionOperator、jordan_winger、
real等函数：
def q_ham_producer(geometry, basis, charge, multiplicity, fermion_transform):
    mol = MolecularData(geometry=geometry,
                        basis=basis,
                        charge=charge,
                        multiplicity=multiplicity,
                        data_directory='./baocun')
    py_mol = run_pyscf(mol, run_scf=1, run_ccsd=1, run_fci=1)

    # Get fermion hamiltonian
    molecular_hamiltonian = py_mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(
        *molecular_hamiltonian.n_body_tensors.values())
    ham_hiq = get_fermion_operator(inter_ops)

    # Get qubit hamiltonian for a given mapping
    if fermion_transform == 'jordan_wigner':
        q_ham = Transform(ham_hiq).jordan_wigner()
        q_ham.compress()
        # print(q_ham)
    elif fermion_transform == 'bravyi_kitaev':
        q_ham = Transform(ham_hiq).bravyi_kitaev()
        q_ham.compress()
        # print(q_ham)

    return (py_mol.n_qubits, py_mol.n_electrons, py_mol.hf_energy,
            py_mol.ccsd_energy, py_mol.fci_energy, q_ham.real)
-（2）修改single和double激发：适当调整输入门参数，将single部分加入两个H控制门
将RY门的参数改为0.5*pai,或者尝试其他快速门分解：
    def _single_qubit_excitation_circuit(self, i, k, singles_counter):
        """
        Implement circuit for single qubit excitation.
        k: creation
        """
        circuit_singles = Circuit()
        circuit_singles += CNOT(i, k)
        circuit_singles += H.on(k,i)
        circuit_singles += RY({f'p_{singles_counter}':0.5*np.pi}).on(k, i)
        circuit_singles += H.on(k,i)
        circuit_singles += CNOT(i, k)
2.再通过梯度下降进行参数优化
 使用get_expectation_with_grad、minimize等函数
self.total_pqc=Simulator(self.backend,self.n_qubits).\
        get_expectation_with_grad(self.sparsed_q_ham,self.total_circuit)
res=minimize(func,
                     self.n_paras,
                     args=(self.total_pqc,),
                     method='bfgs',
                     jac=True,
                     tol=1e-3)  

## 项目代码结构

```bash
.   				<--根目录，提交时将整个项目打包成`hackathon02_队长姓名_队长联系电话.zip`
├── eval.py			<--举办方用来测试选手的模型的脚本,可以作为个人测试使用
├── readme.md		<--说明文档
├── src				<--源代码目录，选手的所有开发源代码都应该放入该文件夹类
│   ├── main.py		<--参考程序模型范例，
│   ├── LiH.hdf5	<--分子1数据文件
│   └── CH4.hdf5	<--分子2数据文件
```

##项目总结与未来改进方向：

1.创新点：用Qubit_excitation代替fermion_excitation。使速度更快，且能达到所要求的化学精度。
2.尝试更好的编码：性能优于jordan_winger
3.更好的single、double的qubit_excitation激发方式，更快且能达到化学精度要求的
single、double线路。

## 参考文献

[1] McArdle, S., Endo, S., Aspuru-Guzik, A., Benjamin, S. C., & Yuan, X. (2020). Quantum computational chemistry. Reviews of Modern Physics, 92(1), 015003.
[2] Li, Y., Hu, J., Zhang, X., Song, Z., & Yung, M. (2019). Variational Quantum Simulation for Quantum Chemistry. Advanced Theory and Simulations, 2(4), 1800182.
[3] Kandala, A., Mezzacapo, A., Temme, K. et al. Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. Nature 549, 242–246 (2017).
[4] Yordanov Y S, Arvidsson-Shukur D R M, Barnes C H W. Efficient quantum circuits for quantum computational chemistry[J]. Physical Review A, 2020, 102(6): 062612.