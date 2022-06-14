# hackathon03(Fenice)

第四届·2022量子计算黑客松全国大赛

[决赛挑战题：3比特幺正算符重构](https://competition.huaweicloud.com/information/1000041660/schedule?track=111)

## 环境要求

- MindQuantum >= 0.5
- MindSpore >= 1.6

## 脚本说明

### 程序结构

```text

├── eval.py       # 判分
├── main.py       # 题解
├── solution.py   # 题解模型
├── encoder_circuit.py
├── ansatz_circuit.py       # 测试线路
├── ansatz_circuit_qsd.py   # 量子香农分解
├── ansatz_circuit_qsd_c.py
├── ansatz_circuit_kgd.py   # 三比特量子门分解
├── layer.py
├── predict.py    # 生成`test_y.npy`
├── testcc.py     # 生成`real_test_y.npy`
├── to_acc.py     # 计算acc
├── utils.py      # 常用函数
├── weights_J49U26B02.npy   # 最终提交模型参数
├── test_y.npy              # 最终提交模型输出结果
├── real_test_y.npy         # 使用`testcc.py`输出的测试集精确解
├── train_x.npy             # 训练集x
├── train_y.npy             # 训练集y
└── test_x.npy              # 测试集x

```

### 下载及部署

1.将 train_x.npy、train_y.npy、test_x.npy 补充完整

2.按顺序执行：

    a. main.py

    b. predict.py

    c. eval.py

## 结果

得分：1
