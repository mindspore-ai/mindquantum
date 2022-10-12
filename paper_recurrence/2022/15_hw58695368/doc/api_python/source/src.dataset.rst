src.dataset
==================

数据集预处理。

|

.. py:function:: src.dataset.build_dataset(nqubits, data_dir, n=None)

    预处理一维横场Ising模型量子数据集，可插值。

    参数：
        - **n_qubits** (int) - 量子比特数。
        - **data_dir** (str) - 数据集文件路径。
        - **n** (int) - 每两个数据之间插值数量。

    返回：
        Circuit，编码量子线路。list，量子线路参数列表。numpy.ndarray，量子线路参数值。numpy.ndarray，标签。

|

.. py:function:: src.dataset.tfi_chain(nqubits, data_dir)

    预处理一维横场Ising模型量子数据集。

    参数：
        - **n_qubits** (int) - 量子比特数。
        - **data_dir** (str) - 数据集文件路径。

    返回：
        Circuit，编码量子线路。list，量子线路参数列表。numpy.ndarray，量子线路参数值。numpy.ndarray，标签。
