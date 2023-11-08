简体中文|[English](README.md)

# Dense ODE-Net 求解 薛定谔方程

## 概述:
薛定谔方程描述了封闭量子系统的演化。 
最近的研究表明，现代神经网络的结钩如残差连接与微分方程的离散结构有相似之处：
[Bridging Deep Architectures and Numerical Differential Equations](https://arxiv.org/pdf/1710.10121.pdf); 
[Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366.pdf). 
神经网络不同层之间的连接与微分方程的离散结钩有着紧密的联系，如残差连接等同于恒等映射。
在ResNet的基础上，DenseNet引入了更密集的Dense Connections, 能够获得更加复杂的网络拓扑结构。
受这些研究的启发，这个demo为ODE-Net引入可学习的Dense Connections，用于求解薛定谔方程的正问题。
关于Dense ODE-Net更详细的介绍参见 [train_中文版](train_CN.ipynb)。
注：本案例中的哈密顿量均厄密且不含时，对于非厄密哈密顿量如PT对称的哈密顿量，本案例暂不考虑。

* ## 训练后的Dense Weights:
    >```
    > -- Current Dense Weight
    > [[0.         0.38127717 0.525522   0.546216   1.        ]
    > [0.         0.7565703  0.77493346 0.8735501  0.768293  ]
    > [0.         0.         0.8958091  0.85228485 0.25397432]
    > [0.         0.         0.         0.90918434 0.21371573]
    > [0.         0.         0.         0.         0.0413048 ]]
    >```
    
* ## 随机哈密顿量预测精度:
  * ![test_acc](images/accuracies_1.png)

## 快速开始
* ### 快速测试 1: 已经训练好的ckpt文件在 `./checkpoints` 目录下，文件名为 `dense_ode_net_step20.ckpt`. 如果想通过该权重文件测试，可以在终端调用 `quick_test.py` 脚本。
    >```
    > python quick_test.py --config_file_path ./config.yaml --device_target CPU --device_id 0 --mode PYNATIVE --save_graphs False --save_graphs_path ./graphs
    >```

* ### 快速测试 1.1: 或者在终端进入`./scripts`目录，调用 `quick_test.sh` shell 脚本
    >```
    > cd {PATH}/Qdynamics/scripts
    > bash quick_test.sh
    >```

* ### 训练方式 1: 在终端调用 `train.py`脚本
    >```
    > python train.py --config_file_path ./config.yaml --device_target CPU --device_id 0 --mode PYNATIVE --save_graphs False --save_graphs_path ./graphs
    >```

* ### 训练方式 1.1: 在终端进入`./scripts`目录，调用 `train.sh`shell脚本
    >```
    > cd {PATH}/Qdynamics/scripts
    > bash train.sh
    >```

* ### 训练方式 2: 运行 Jupyter Notebook
    您可以使用[train_中文版](train_CN.ipynb) 或 [train_英文版](train.ipynb) Jupyter Notebook 逐行运行训练。

* ### Requirements:
    详见[requirements](requirements.txt)

## 贡献者
* ### 止三
* ### 电子邮箱： 762598802@qq.com