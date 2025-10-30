FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    git \
    && apt-get clean

# 设置 python / pip 默认指向 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 安装 JupyterLab 已经配置了vscode jupyter server 可以运行
RUN pip install --upgrade pip && pip install jupyterlab 
RUN pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.6.0/MindSpore/unified/x86_64/mindspore-2.6.0-cp310-cp310-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://repo.huaweicloud.com/repository/pypi/simple
RUN pip install mindquantum

# 创建工作目录
WORKDIR /root/
