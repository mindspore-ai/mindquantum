# A hybrid classical-quantum approach for multi-class classification

## 项目介绍

利用变分量子线路对三组数据集(Iris dataset, Bankonte Authentication dataset 和 Wireless Indoor Localization dataset)进行分类

## 主要结果

 1. 论文中采用了一种新的编码方式和传统的Amplititude embedding encoding编码方式做了对比。
 2. 论文中的量子线路加入辅助比特，将数据的编码比特和辅助比特纠缠在一起。
 2. 代码结果展示了 Iris dataset 的训练结果，没有展示了 Bankonte Authentication dataset 的训练结果。因为数据集 Bankonte Authentication dataset 数据太大，训练过程中用cloudide中间程序中断了，没有训练出来。


## 总结：

1. 与传统的Amplititude embedding encoding 编码方式相比，论文中给出的量子编码方式所需的量子比特更少，但是要对训练数据做预处理。
2. 文章中量子线路里增加了辅助比特，通过测量辅助比特泡利算符Z的期望值来作为分类的结果， 能更好的实现多分类问题 。


邮箱地址：tang13419@gmail.com
