# 代码说明

- ### `RP(rotation_angle)`
	- 返回一个随机单比特Pauli旋转门，等概率地从$R_X, R_Y, R_Z$中选一个。
	- 参数说明
		- rotation_angle -- 表示Pauli旋转门的旋转角度。            

- ### `bpansatz(num_qubits,number_layers)`
	- 返回一个随机量子线路，与论文原文一致（参考main.py或者上一层目录中的readme.md）。初态由$R_Y(\pi/4)$制备，每个单层包括一系列含参的随机Pauli旋转门（RP）与依次连接相邻比特位的无参数控制Z门。
	- 参数说明
		- num_qubits -- 表示线路的比特数；
		- num_layers -- 表示线路的层数。

- ### `get_var_partial_exp(circuit, hamiltonian, number_of_attempts = 10, error_request = 0.2)`
	- 返回含参线路的期望值的偏导数方差，使用`get_expectation_with_grad`只对线路中第一层、第一个比特上的含参量子门求导，通过逐次迭代的采样方式，每次迭代样本容量会翻一倍。每次迭代，程序会自动进行审敛，并且输出基于样本的相对误差。当误差满足要求会停止迭代，返回结果。
	- 参数说明
		- circuit -- 用于求导的含参数量子线路；
		- hamiltonian -- 在线路尾端测量时使用的哈密顿量；
		- number_of_attempts -- 初次采样的样本容量，并不是最终样本容量，程序会从这个样本容量出发，不断增加要本容量，直到误差满足要求。默认值为10；
		- error_request -- 误差要求，必须介于0-1之间。默认值为0.2。