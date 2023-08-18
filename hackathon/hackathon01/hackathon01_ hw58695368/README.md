# hackathon01(Fenice)

第四届·2022量子计算黑客松全国大赛

[赛题1：利用量子经典混合的神经网络来完成手写字体识别任务](https://competition.huaweicloud.com/information/1000041660/circumstance?track=111)

## 环境要求

- MindQuantum >= 0.5
- MindSpore >= 1.6

## 脚本说明

### 程序结构

```text

├── eval.py
├── src
│   ├── main.py              # 题解
│   ├── hybrid.py            # 框架
│   ├── qclib.py             # 量子线路组件
│   ├── qcexlib.py           # 量子线路库
│   ├── checkdataset.py      # 数据检查
│   ├── __train.py           # 训练
│   ├── model.ckpt           # 模型参数
│   ├── test.pkl             # 测试集
│   ├── train_statistics.pkl # 训练集统计结果
│   └── train.npy            # 训练集
└── test.npy                 # 测试集

```

### 下载及部署

1.将 train.npy、test.npy 补充完整

2.执行 eval.py

## 结果

得分：0.9375
