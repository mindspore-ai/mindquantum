# MindQuantum Docker安装指南

<!-- TOC --->

- [Docker方式安装MindQuantum](#docker方式安装mindquantum)
    - [获取MindSpore镜像](#获取mindspore镜像)
    - [运行MindSpore镜像](#运行mindspore镜像)
    - [验证Mindspore是否安装成功](#验证mindspore是否安装成功)
    - [在Docker容器内安装Mindquantum](#在docker容器内安装mindquantum)
    - [验证MindQuantum是否安装成功](#验证mindquantum是否安装成功)

<!-- TOC --->

## Docker方式安装MindQuantum

本文档介绍如何使用Docker方式快速安装MindQuantum。首先需要通过Docker方式安装MindSpore，流程介绍在[MindSpore的官方网站](https://www.mindspore.cn/install)，以下将重复这部分的内容。

### 获取MindSpore镜像

对于CPU后端，可以直接使用以下命令获取最新的稳定镜像：

```shell
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:{tag}
```

其中{tag}为x.y.z或者devel或者runtime，具体请参考MindSpore的docker[安装指南](https://www.mindspore.cn/install)

### 运行MindSpore镜像

执行以下命令启动Docker容器实例：

```shell
docker run -it swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:{tag} /bin/bash
```

其中{tag}与上述tag保持一直。

### 验证Mindspore是否安装成功

- 如果你安装的是指定版本x.y.z的容器，执行如下步骤。

按照上述步骤进入MindSpore容器后，测试Docker是否正常工作，请运行下面的Python代码并检查输出：

```python
import numpy as np
import mindspore as ms
from mindspore import set_context, ops, Tensor

set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

x = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
y = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
print(ops.add(x, y))
```

代码成功运行时会输出：

```python
[
    [
        [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
        [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
        [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
    ]
]
```

至此，你已经成功通过Docker方式安装了MindSpore CPU版本。

### 在Docker容器内安装Mindquantum

1. 进入Docker容器

    ```shell
    docker exec -it {docker_container} /bin/bash
    ```

    其中{docker_container} 是docker容器的id或者名字

2. 选择编译安装或者pip安装

    **编译安装：**

    ```shell
    git clone https://gitee.com/mindspore/mindquantum.git
    cd ~/mindquantum
    python setup.py install --user
    ```

    **pip安装：**

    ```shell
    pip install https://hiq.huaweicloud.com/download/mindquantum/newest/linux/mindquantum-master-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

### 验证MindQuantum是否安装成功

```shell
python -c 'import mindquantum'
```
