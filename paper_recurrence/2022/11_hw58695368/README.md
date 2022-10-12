<TOC>

# MBE

Variational Quantum Optimization with Multi-Basis Encodings(MBE)。

[复现论文](https://arxiv.org/abs/2106.13304)："Variational Quantum Optimization with Multi-Basis Encodings"

## 数据集

- 自行输入。

## 环境要求

- 硬件（CPU）
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
    - [MindQuantum](https://gitee.com/mindspore/mindquantum)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindQuantum教程](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/index.html)

- 第三方库

```bash
pip install numpy
pip install scipy
```

## 快速入门

1. 执行测试用例。按照如下步骤启动测试：

   ```bash
   # 测试
   python eval.py
   ```

## 脚本说明

### 脚本和样例代码

```shell
mbe
├── README.md                           # 模型说明文档
├── requirements.txt                    # 依赖说明文件
├── src                                 # 模型定义源码目录
│   ├── maxcut.py                       # 优化算法定义
│   ├── ansatz_mpo.py                   # 量子拟设定义
│   ├── mbe_loss.py                     # 损失函数定义
│   ├── layer.py                        # 可训练层定义
│   └── dataset.py                      # 数据集处理定义
└── eval.py                             # 算法测试脚本
```

## 测试过程

### 测试

- 运行`eval.py`测试算法。

## 性能

### 测试性能

| 参数                  | MBE                    | MBE(grad)              | MBE(grad)              | MBE(grad)              |
| --------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| 模型版本              | V1                     | V2                     | V2                     |V2                      |
| 资源                  | Ubuntu 18.04.5 LTS     | Ubuntu 18.04.5 LTS     | Ubuntu 18.04.5 LTS     | Ubuntu 18.04.5 LTS     |
| 上传日期              | 2022-06-27             | 2022-07-02             | 2022-07-02             | 2022-07-02             |
| MindSpore版本         | 1.6.0                  | 1.6.0                  | 1.6.0                  | 1.6.0                  |
| MindQuantum版本       | 0.5.0                  | 0.5.0                  | 0.5.0                  | 0.5.0                  |
| 数据集                | 自设dataset1           | 自设dataset1           | 自设dataset2           | 自设dataset3           |
| 优化器                | Nelder-Mead            | BFGS                   | BFGS                   | BFGS                   |
| 损失函数              | MBE损失函数            | MBE损失函数            | MBE损失函数            | MBE损失函数            |
| 线路深度L             | 7                      | 7                      | 4                      | 4                      |
| 测试结果              | success                | success                | success                | success                |
| 脚本                  | [link](https://gitee.com/NoEvaa/zero/tree/master/NPark/NGPark/mindspore/mbe/)  |

| 参数                  | MBE-parallel(grad)       | MBE-parallel(grad)       |
| --------------------- | ------------------------ | ------------------------ |
| 模型版本              | V2                       | V2                       |
| 资源                  | Ubuntu 18.04.5 LTS       | Ubuntu 18.04.5 LTS       |
| 上传日期              | 2022-07-02               | 2022-07-02               |
| MindSpore版本         | 1.6.0                    | 1.6.0                    |
| MindQuantum版本       | 0.5.0                    | 0.5.0                    |
| 数据集                | 自设dataset1和dataset2   | 自设dataset2和dataset3   |
| 优化器                | BFGS                     | BFGS                     |
| 损失函数              | MBE损失函数              | MBE损失函数              |
| 线路深度L             | 4                        | 4                        |
| 测试结果              | success                  | success                  |
| 脚本                  | [link](https://gitee.com/NoEvaa/zero/tree/master/NPark/NGPark/mindspore/mbe/)  |

## 随机情况说明

在maxcut.py中进行权重随机初始化。

### 贡献者

* [NoEvaa](https://gitee.com/NoEvaa)

## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。