# MindQuantum Docker Installation GUide

<!-- TOC --->

- [Installing MindQuantum by Docker](#installing-mindquantum-by-docker)
    - [Obtaining the MindSpore Image](#obtaining-the-mindSpore-image)
    - [Running the MindSpore Image](#running-the-mindspore-image)
    - [MindSpore Installation Verification](#mindspore-installation-verification)
    - [Installing MindQuantum Inside a Docker Container](#installing-mindquantum-inside-a-docker-container)
    - [MindQuantum Installation Verification](#mindquantum-installation-verification)

<!-- TOC --->

## Installing MindQuantum by Docker

This document describes how to use Docker to quickly install MindQuantum. First, you need to install MindSpore through Docker. The process is introduced on [MindSpore's official website](https://www.mindspore.cn/install/en). This part will be repeated below.

### Obtaining the MindSpore Image

For the CPU backend, first refer to the MindSpore Docker [installation guide](https://www.mindspore.cn/install/en) to confirm the currently recommended official image name and tag, and then run:

```shell
docker pull {mindspore_image}:{tag}
```

Here `{mindspore_image}` and `{tag}` should follow the official MindSpore installation guide.

### Running the MindSpore Image

Execute the following command to start the Docker container instance:

```shell
docker run -it {mindspore_image}:{tag} /bin/bash
```

Here `{mindspore_image}` and `{tag}` should match the previous step.

### MindSpore Installation Verification

- If you are installing a container of the specified version x.y.z.

After entering the MindSpore container according to the above steps, to test whether Docker is working properly, please run the following Python code and check the output:

```python
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor

ms.set_device("CPU")
ms.set_context(mode=ms.PYNATIVE_MODE)

x = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
y = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
print(ops.add(x, y))
```

When the code runs successfully, it will output:

```text
[[[[2., 2., 2., 2.],
   [2., 2., 2., 2.],
   [2., 2., 2., 2.]],

  [[2., 2., 2., 2.],
   [2., 2., 2., 2.],
   [2., 2., 2., 2.]],

  [[2., 2., 2., 2.],
   [2., 2., 2., 2.],
   [2., 2., 2., 2.]]]]
```

At this point, you have successfully installed the MindSpore CPU version by Docker.

### Installing MindQuantum Inside a Docker Container

1. Enter the Docker container.

    ```shell
    docker exec -it {docker_container} /bin/bash
    ```

    In which, {docker_container} is the id or name of the docker container.

2. Choose to install by pip or by compiling.

    **Install by compiling:**

    ```shell
    git clone https://atomgit.com/mindspore/mindquantum.git
    cd ~/mindquantum
    python setup.py install --user
    ```

    **Install by pip:**

    ```shell
    pip install mindquantum
    ```

### MindQuantum Installation Verification

```bash
python -c 'import mindquantum'
```
