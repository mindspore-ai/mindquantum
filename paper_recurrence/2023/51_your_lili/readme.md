## 项目简介

基于 Mindquantum，实现论文 "Simulation of Modular Exponentiation Circuit for Shor’s Algorithm in Qiskit" 
中提出的加法器、减法器、模加法器、模乘法器及模指数器等线路。

## 项目目录

项目结构如下：

```text
.
├── images              # main.ipynb使用介绍相关图片
├── main.ipynb          # 代码使用案例
├── readme.md           # 本介绍文档
└── src/                # 包含项目完整代码
    ├── circuit.py      # 模加法器、模指数器等
    ├── demo.py         # 测试使用模加法器等
    └── utils.py        # 包含一些经典和量子比特输入输出转换等
```

## 参考文献

[H. T. Larasati and H. Kim, "Simulation of Modular Exponentiation Circuit for Shor's Algorithm in Qiskit," 2020 14th International Conference on Telecommunication Systems, Services, and Applications, TSSA, Bandung, Indonesia, 2020, pp. 1-7, doi: 10.1109/TSSA51342.2020.9310794.](https://ieeexplore.ieee.org/document/9310794)

