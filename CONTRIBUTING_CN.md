# MindSpore Quantum贡献指南

[View English](https://gitee.com/mindspore/mindquantum/blob/master/CONTRIBUTING.md)

- [MindSpore Quantum贡献指南](#mindspore-quantum贡献指南)
    - [贡献者许可协议](#贡献者许可协议)
    - [量子计算简介](#量子计算简介)
    - [安装与开发](#安装与开发)
        - [安装](#安装)
            - [pip安装](#pip安装)
            - [源码安装](#源码安装)
            - [验证是否成功安装](#验证是否成功安装)
        - [编译](#编译)
        - [开发](#开发)
    - [快速入门](#快速入门)
    - [代码结构](#代码结构)
    - [单元测试](#单元测试)
    - [编写文档](#编写文档)
    - [贡献流程](#贡献流程)
        - [代码风格](#代码风格)
        - [Fork-Pull开发模型](#fork-pull开发模型)
        - [报告Issue](#报告issue)
        - [提交PR](#提交pr)
        - [本地代码自检](#本地代码自检)

## 贡献者许可协议

向MindSpore Quantum社区提交代码之前，您需要阅读并签署《贡献者许可协议（CLA）》。

个人贡献者请参见[ICLA在线文件](https://www.mindspore.cn/icla)。

## 量子计算简介

随着摩尔定律的逐渐失效，集成电路板制程提升和芯片性能提升愈加困难，算力到达瓶颈期。经典计算机的架构已近极限，在大模型算力要求越加突出的当下，`量子计算`这种新型的技术领域逐渐得到大众的关注。

量子是个物理学的概念，意思是最小化、不可再分的基本单位，最小单位即是量子；不可再分的概念称之为量子化。可以用于描述微观物理世界中的粒子特性，例如原子，电子。“量子”来自于拉丁语*quantum*，意译是“一定数量的物质”。

**量子计算是一项前沿技术，使用量子力学的规律调控量子信息单元进行计算**。简言之，操纵量子比特用于计算机。当前，人们使用手机、电脑、其他媒介刷到本文章，其软件底层都是0和1数据流，对应的硬件控制就是高低电平；而量子计算操控的是量子比特，具备量子态特性的微观粒子。

量子计算的应用相当广泛，在密码解析、交通运输问题（组合优化数学问题），材料化工、生物医学（化学分子能量模拟计算），人工智能（量子-机器学习），新型半导体（量子芯片）等领域有着广泛应用。
$$
量子态\\|\varphi>=\alpha |0>+\beta |1>
$$

## 安装与开发

### 安装

安装MindSpore

请根据MindSpore官网[安装指南](https://www.mindspore.cn/install)，安装1.4.0及以上版本的MindSpore。

MindSpore是一种适用于端边云场景的新型开源深度学习训练/推理框架。 MindSpore提供了友好的设计和高效的执行，旨在提升数据科学家和算法工程师的开发体验。

#### pip安装

- 安装MindQuantum

```bash
pip install mindquantum
```

#### 源码安装

1. 从代码仓下载源码

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindquantum.git
    ```

2. 编译MindQuantum

    **Linux系统**下请确保安装好CMake >= 3.18.3，然后运行如下命令：

    ```bash
    cd ~/mindquantum
    bash build.sh --gitee
    ```

    这里 `--gitee` 让脚本从gitee代码托管平台下载第三方依赖。如果需要编译GPU版本，请先安装好 CUDA 11.x，和对应的显卡驱动，然后使用`--gpu`参数，执行如下编译指令：

    ```bash
    cd ~/mindquantum
    bash build.sh --gitee --gpu
    ```

    **Windows系统**下请确保安装好MinGW-W64和CMake >= 3.18.3，然后运行如下命令：

    ```bash
    cd ~/mindquantum
    build.bat /Gitee
    ```

    **Mac系统**下请确保安装好openmp和CMake >= 3.18.3，然后运行如下命令：

    ```bash
    cd ~/mindquantum
    bash build.sh --gitee
    ```

3. 安装编译好的whl包

    进入output目录，通过`pip`命令安装编译好的mindquantum的whl包。

#### 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindquantum'`，则说明安装成功。

```bash
python -c 'import mindquantum'
```

### 编译

MindQuantum提供**编译出包**和**本地编译**两种方法

1.编译出包，如果用户需要适配不同的系统环境、python版本，则可以将源码编译成wheel包，再安装，详细流程可参考上文[源码安装](# 源码安装)。

2.本地编译，如果用户新增或修改部分代码，需要快速验证是否生效，则可以使用本地编译，将C++代码编译成*.so文件，再使用Python调用，并设置环境变量，不需要重新编译出包，再卸载并重新安装，方便调试验证。

从代码仓下载源码

```bash
cd ~
git clone https://gitee.com/mindspore/mindquantum.git
```

- **Linux**系统，依赖于CMake >= 3.18.3，本地编译。

  `--gitee`参数是指定脚本从gitee代码托管平台下载第三方依赖；`export`命令是添加mindquantum源码路径到PYTHONPATH环境变量里。

  ```bash
  cd ~/mindquantum

  bash build_locally.sh --gitee
  export PYTHONPATH=`pwd`$PYTHONPATH
  ```

- **Windows**系统，依赖于MinGW-W64和CMake >= 3.18.3，本地编译。

  ```bash
  cd ~/mindquantum
  build_locally.bat /Gitee -G 'MinGW Makefiles'
  set PYTHONPATH=%cd%;%PYTHONPATH%
  ```

- **Mac**系统，依赖于openmp和CMake >= 3.18.3，本地编译。

  ```bash
  cd ~/mindquantum
  bash build_locally.sh --gitee
  export PYTHONPATH=`pwd`$PYTHONPATH
  ```

- 系统依赖组件安装
    - GCC-GNU编译器套件，安装说明

      1.Linux和MacOS安装

      1.1Linux/Ubuntu/Centos系统和MacOS系统会自带`gcc`，版本满足编译MindQuantum。

      1.2使用命令安装gcc

      ```shell
      # Ubuntu 安装
      apt-get update
      apt-get install gcc
      apt-get install build-essential

      # Centos 安装
      yum install gcc gcc-c++

      # 输出gcc版本，验证安装成功
      gcc --version
      ```

      2.Window安装

      2.1 Window的C/C++编译器套件通常推荐Mingw-w64，可用于编译和运行Window应用和DLL文件。

      2.2 进入[Mingw-w64安装包](https://sourceforge.net/projects/mingw-w64/files/)页面，下载离线安装包。这里推荐**x86_64-posix-seh**版本，对应x64架构。

      2.3 本地双击打开安装包，按照提示，进行安装。（需要注意，写入环境变量）

      2.4安装完毕，打开cmd终端，输入命令

      ```shell
      C:\Users\xx>gcc --version

      gcc (x86_64-win32-seh-rev0, Built by MinGW-W64 project) 8.1.0
      Copyright (C) 2018 Free Software Foundation, Inc.
      ```

      遇到bug，可发issue，或谷歌百度，参考[csnd教程](https://blog.csdn.net/jiqiren_dasheng/article/details/103775488)。

    - CMake-跨平台的开源构建程序，安装说明

      1.进入[CMake](https://cmake.org/)官网，根据系统选择合适的版本的安装包。

      Windows系统推荐**cmake-3.2\*-windows-x86_64.msi**，

      Macos系统推荐**cmake-3.2\*-macos-universal.dmg**，

      Linux系统推荐**cmake-3.2*-linux-x86_64.sh**。

      2.本地双击打开CMake安装包，依照提示安装，注意需要将CMake添加进用户变量中，最后点击`Finish`，安装完毕。

      3.打开终端，在命令行中输出命令，输出版本号即表示成功。

      ```shell
      cmake --version
      >>> cmake version 3.24.2
      ```

### 开发

mindquantum主要使用C++和python进行开发，核心计算单元使用C/C++实现，上层接口及周边模块使用Python实现

开发流程主要分成2类，开发新功能，修复缺陷bug：

- 开发新功能，进入MindQuantum的issue页面，提交issue，编写新功能描述、类别、实现方法等。与MindQuantum开发团队交流，确认该功能是否必要。本地着手实现，编写代码，实现功能，编写相应的测试用例，相应的说明文档。提交PR，待代码通过审查后合入主分支，新功能开发完毕。

- 修复缺陷bug，进入MindQuantum的issue页面，阅读未关闭的issue，认领issue解决问题。或者平时使用MindQuantum遇到bug，欢迎提交issue，帮助完善MindQuantum功能模块。

## 快速入门

- 在[Gitee](https://gitee.com/mindspore/mindquantum)上fork mindquantum代码仓。
- 参见[README_CN.md](https://gitee.com/mindspore/mindquantum/blob/master/README_CN.md)了解项目信息和构建说明。
- 初体验-搭建参数化量子线路
  使用 `mindquantum`搭建包括H门、RX门和RY门的量子线路，并得到量子态

```python
from mindquantum import *
import numpy as np

encoder = Circuit().h(0).rx({'a0': 2}, 0).ry('a1', 1)
print(encoder)
print(encoder.get_qs(pr={'a0': np.pi / 2, 'a1': np.pi / 2}, ket=True))
```

运行上述代码，将会输出量子线路和末态

```bash
      ┏━━━┓ ┏━━━━━━━━━━┓
q0: ──┨ H ┠─┨ RX(2*a0) ┠───
      ┗━━━┛ ┗━━━━━━━━━━┛
      ┏━━━━━━━━┓
q1: ──┨ RY(a1) ┠───────────
      ┗━━━━━━━━┛
-1/2j¦00⟩
-1/2j¦01⟩
-1/2j¦10⟩
-1/2j¦11⟩
```

## 代码结构

- [`ccsrc`](https://api.gitee.com/mindspore/mindquantum/tree/master/ccsrc) 核心运算模块，使用C/C++实现
- [`cmake`](https://api.gitee.com/mindspore/mindquantum/tree/master/cmake) cmake编译配置信息
- [`docs`](https://api.gitee.com/mindspore/mindquantum/tree/master/docs) MindQuantum API文档
- [`mindquantum`](https://api.gitee.com/mindspore/mindquantum/tree/master/mindquantum) MindQuantum量子计算模块，使用Python实现
    - [`mindquantum.dtype`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.dtype.html#module-mindquantum.dtype) MindQuantum 数据类型模拟。
    - [`mindquantum.core`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.html#module-mindquantum.core) MindQuantum的核心特性(eDSL)。
        - [`gata`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/mindquantum.core.gates.html) 量子门模块，提供不同的量子门。
        - [`circuit`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/mindquantum.core.circuit.html) 量子线路模块，可以轻松地搭建出符合要求的量子线路，包括参数化量子线路。
        - [`operators`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/mindquantum.core.operators.html) MindQuantum 算子库
        - [`parameterresolver`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/mindquantum.core.parameterresolver.html) 参数解析器模块，用于声明使用到的参数。
    - [`mindquantum.simulator`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.simulator.html#module-mindquantum.simulator) 模拟量子系统演化的量子模拟器。
    - [`mindquantum.framework`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.framework.html#module-mindquantum.framework) 量子神经网络算子和cell。
    - [`mindquantum.algorithm`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.html#module-mindquantum.algorithm) 量子算法。
        - [`compiler`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mindquantum.algorithm.compiler.html) 量子线路编译模块
        - [`library`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mindquantum.algorithm.library.html) 常用算法模块
        - [`nisq`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mindquantum.algorithm.nisq.html) NISQ算法
        - [`error_mitigation`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mindquantum.algorithm.error_mitigation.html) 误差缓解模块
        - [`mapping`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mindquantum.algorithm.mapping.html) 比特映射模块
    - [`mindquantum.device`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.device.html#module-mindquantum.device) MindQuantum 硬件模块。
    - [`mindquantum.io`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.io.html#module-mindquantum.io) MindQuantum的输入/输出模块。
    - [`mindquantum.engine`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.engine.html#module-mindquantum.engine) MindQuantum引擎模块。
    - [`mindquantum.utils`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.utils.html#module-mindquantum.utils) 实用工具。
- [`mindquantum_config`](https://api.gitee.com/mindspore/mindquantum/tree/master/mindquantum_config) 项目配置信息
- [`scripts`](https://api.gitee.com/mindspore/mindquantum/tree/master/scripts) 编译依赖工具更新脚本
- [`tests`](https://api.gitee.com/mindspore/mindquantum/tree/master/tests) MindQuantum 单元测试，基于Pytest，使用Python编写
- [`third_party`](https://api.gitee.com/mindspore/mindquantum/tree/master/third_party) MindQuantum编译依赖的第三方开源包
- [`tutorials`](https://api.gitee.com/mindspore/mindquantum/tree/master/tutorials) MindQuantum 教程教案，使用jupyter可以直接运行，在[mindspore官方文档](https://mindspore.cn/mindquantum/docs/zh-CN/master/index.html)可阅读。

## 单元测试

mindquantum基于pytest编写单元测试用例，建议开发者在实现一个新的功能、模块后，编写对应的单元测试用例，保证功能正常

## 编写文档

编写文档的说明指南，MindQuantum 有两种主要类型的文档：

- 面向用户的文档：这些是用户在[MindSpore Quantum网站](https://mindspore.cn/mindquantum/docs/zh-CN/master/index.html)上看到的文档，包括线路构建，模拟器，算法实现等教程教案，有助于快速入手并应用MindQuantum，也有利于学习量子计算算法。
- 面向开发人员的文档：面向开发人员的文档分布在代码库`MindQuanntum/docs`中。如果有兴趣添加新的开发人员文档，请阅读wiki 上的此页面，了解最佳实践，并在编写代码后及时书写注释，API是通过抽取代码中的注释信息整理而成。

## 贡献流程

### 代码风格

请遵循此风格，以便MindSpore Quantum团队审查、维护和开发。

- 编码指南

  MindSpore Quantum社区使用[Python PEP 8 编码风格](https://pep8.org/)和[谷歌C++编码风格](http://google.github.io/styleguide/cppguide.html)。建议在IDE中安装以下插件，用于检查代码格式：[CppLint](https://github.com/cpplint/cpplint)、[CppCheck](http://cppcheck.sourceforge.net)、[CMakeLint](https://github.com/cmake-lint/cmake-lint)、[CodeSpell](https://github.com/codespell-project/codespell)、[Lizard](http://www.lizard.ws)、[ShellCheck](https://github.com/koalaman/shellcheck)和[PyLint](https://pylint.org)。
- 单元测试指南

  MindSpore社区使用Python单元测试框架[pytest](http://www.pytest.org/en/latest/)。注释名称需反映测试用例的设计意图。
- 重构指南

  我们鼓励开发人员重构我们的代码，以消除[代码坏味道](https://zh.wikipedia.org/wiki/%E4%BB%A3%E7%A0%81%E5%BC%82%E5%91%B3)。所有代码都要符合编码风格和测试风格，重构代码也不例外。无注释的代码行（nloc）的[Lizard](http://www.lizard.ws)阈值为100，圈复杂度（cnc）的阈值为20。当收到Lizard警告时，必须重构要合并的代码。
- 文档指南

  我们使用MarkdownLint来检查Markdown文档格式。MindSpore CI基于默认配置修改了以下规则。
    - MD007（无序列表缩进）：参数**indent**设置为**4**，表示无序列表中的所有内容都需要缩进4个空格。
    - MD009（行尾空格）：参数**br_spaces**设置为**2**，表示行尾可以有0或2个空格。
    - MD029（有序列表的序列号）：参数**style**设置为**ordered**，表示升序。

  有关详细信息，请参见[规则](https://github.com/markdownlint/markdownlint/blob/master/docs/RULES.md)。

### Fork-Pull开发模型

- Fork MindSpore代码仓

  在提交代码至MindSpore项目之前，请确保已fork此项目到您自己的代码仓。MindSpore代码仓和您自己的代码仓之间可能会并行开发，请注意它们之间的一致性。
- 克隆远程代码仓

  如果您想将代码下载到本地计算机，最好使用Git方法：

  ```shell
  # 在Gitee上

  git clone https://gitee.com/{insert_your_forked_repo}/mindquantum.git
  ```

- 本地开发代码。

为避免分支不一致，建议切换到新分支：

```shell
git checkout -b {新分支名称} origin/master
```

以master分支为例，如果MindSpore Quantum需要创建版本分支和下游开发分支，请先修复上游的bug，

再更改代码。

- 将代码推送到远程代码仓。

  更新代码后，以正式的方式推送更新：

```shell
git add .
git status # 查看更新状态。
git commit -m "你的commit标题"
git commit -s --amend # 添加commit的具体描述，可选。
git push origin {新分支名称}
```

- 新建Pr提交到MindSpore Quantum代码仓。

在最后一步中，您需要在新分支和MindSpore Quantum主分支之间拉取比较请求。完成拉取请求后，Jenkins CI将自动设置，进行构建测试。拉取请求应该尽快合并到上游master分支中，以降低合并的风险。

最后一步，您需要在您的代码仓点击`Pull Requests`，新建Pr，选择源分支和目标分支，比较更新后的代码差异。提交Pr请求后，评论区会出现`i-robot`小助手进行操作说明。

如果您是第一次提交Pr，将会出现*mindspore-cla/no*标签，`i-robot`小助手会提示您签署贡献者许可协议。如果已经签署过，则会出现*mindspore-cla/yes*标签。

然后，在评论区输入`/retest`，触发Jenkins CI门禁系统，进行编译构建、代码语法、单元测试等相关测试，保证合入的代码质量。

### 报告Issue

发现问题后，建议以报告issue的方式为项目作出贡献。错误报告应尽量书写规范，内容详尽，感谢您对项目作出的贡献。

报告issue时，请参考以下格式：

- 说明您使用的环境版本（MindSpore、OS、Python等）。

- 说明是错误报告还是功能需求。

- 说明issue类型，添加标签可以在issue板上突出显示该issue。

- 问题是什么？

- 期望如何处理？

- 如何复现？（尽可能精确具体地描述）

- 给审核员的特别说明。

**Issue咨询：**

- **解决issue时，请先评论**，告知他人由您来负责解决该issue。

- **对于长时间未关闭的issue**，建议贡献者在解决该issue之前进行预先检查。

- **如您自行解决了自己报告的issue**，仍需在关闭该issue之前告知他人。

- **如需issue快速响应**，可为issue添加标签。标签详情，参见[标签列表](https://gitee.com/mindspore/community/blob/master/sigs/dx/docs/labels.md)。

### 提交PR

- 在[Gitee](https://api.gitee.com/mindspore/mindquantum/issues)上通过issue提出您的想法。

- 如果是需要大量设计细节的新功能，还应提交设计方案。

- 经issue讨论和设计方案评审达成共识后，在已fork的代码仓开发，并提交PR。

- 任何PR至少需要位2位审批人的LGTM标签。请注意，审批人不允许在自己的PR上添加LGTM标签。

- 经充分讨论后，根据讨论的结果合并、放弃或拒绝PR。

**PR咨询：**

- 避免不相关的更改。

- 确保您的commit历史记录有序。

- 确保您的分支与主分支始终一致。

- 用于修复错误的PR中，确保已关联所有相关问题。

### 本地代码自检

在开发过程中，建议使用pre-push功能进行本地代码自检，可以在本地进行类似CI门禁上Code Check阶段的代码扫描，提高上库时跑门禁的成功率。使用方法请参见[pre-push快速指引](scripts/pre_commit/README_CN.md)。

开发完成后，建议*vscode*或*pycharm*的格式化功能，规范代码风格。
