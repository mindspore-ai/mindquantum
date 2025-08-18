
### 量子误差缓解赛道-高荣浩

论文：[基于单比特校准矩阵与局部 IBU 的测量误差校准方案](readout-高荣浩.md)

`answer.py`：是最终的可运行代码，包含原始ibu，向量化ibu，单比特局部ibu，矩阵求逆法，按需注释即可使用

`answer_test.py`：是测试代码，用来输出各基态在矩阵取逆法和局部 IBU 法下的性能对比

`answer_matrix.py`：是随机矩阵法，通过构造训练集，去训练校准矩阵，可直接运行，不需要通过 `run.py`。有两个可变的方法参数，`method="random_martix"`， `data_method="ground_state"`，表示初始化随机矩阵和使用基态构造的数据，可根据注释选择需要的方法。训练好的矩阵会进行存储

`matrix_picture.py`：是将训练好的矩阵，用于校准基态，查看其性能