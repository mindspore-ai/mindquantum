# MindSpore Quantum Contributing Guidelines

[查看中文](https://gitee.com/mindspore/mindquantum/blob/master/CONTRIBUTING_CN.md)

- [MindSpore Quantum Contributing Guidelines](#mindspore-quantum-contributing-guidelines)
    - [Contributor License Agreement](#contributor-license-agreement)
    - [Quantum Computing Introduction](#quantum-computing-introduction)
    - [Installation and development](#installation-and-development)
        - [Installation](#installation)
            - [Pip installation](#pip-installation)
            - [Source code installation](#source-code-installation)
            - [Verify installation successful](#verify-installation-successful)
        - [Compile](#compile)
        - [Development](#development)
    - [Quick start](#quick-start)
    - [Code structure](#code-structure)
    - [Unit testing](#unit-testing)
    - [Write document](#write-document)
    - [Contribution Workflow](#contribution-workflow)
        - [Code style](#code-style)
        - [Fork-Pull development model](#fork-pull-development-model)
        - [Report issues](#report-issues)
        - [Propose PRs](#propose-prs)
        - [Local code self-test](#local-code-self-test)

## Contributor License Agreement

It's required to sign CLA before your first code submission to MindSpore Quantum community.

For individual contributor, please refer to [ICLA online document](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Ficla) for the detailed information.

## Quantum Computing Introduction

With the gradual failure of Moore's Law, it is more and more difficult to improve the integrated circuit board process and chip performance, and the computing power has reached a bottleneck. The architecture of classical computers is close to the limit, and this new technology field has gradually attracted the attention of the public at a time `quantum computing` when the computing power requirements of large models are becoming more and more prominent.

Quantum is a physical concept, which means the minimum and indivisible basic unit. The minimum unit is quantum; the indivisible concept is called quantization. It can be used to describe the characteristics of particles in the micro-physical world, such as atoms and electrons. "Quantum" comes from the Latin *quantum*, which translates as "a certain amount of matter".

**Quantum computing is a cutting-edge technology, which uses the laws of quantum mechanics to control quantum information units for computing.**。 In short, manipulating quantum bits is used in computers. At present, people use mobile phones, computers and other media to read this article. The underlying software is 0 and 1 data streams, and the corresponding hardware control is high and low levels. Quantum computing manipulates quantum bits, microscopic particles with quantum state characteristics.

Quantum computing is widely used in cryptography analysis, transportation problems (combinatorial optimization mathematical problems), materials and chemical engineering, biomedicine (chemical molecular energy simulation), artificial intelligence (quantum-machine learning), new semiconductors (quantum chips) and other fields.

quantum state
$$
|\varphi>=\alpha |0>+\beta |1>
$$

## Installation and development

### Installation

Install MindSpore

Please install MindSpore version 1.4.0 and above according to MindSpore website [Installation Guide](https://www.mindspore.cn/install/en).

MindSpore is a new open source deep learning training/inference framework that could be used for mobile, edge and cloud scenarios. MindSpore is designed to provide development experience with friendly design and efficient execution for the data scientists and algorithmic engineers, native support for Ascend AI processor, and software hardware co-optimization.

#### Pip installation

- Install MindQuantum

```bash
pip install mindquantum
```

#### Source code installation

- Download the source code from the Gitee

```bash
cd ~
git clone https://gitee.com/mindspore/mindquantum.git
```

- Compile MindQuantum

    Make **Linux system** sure that CMake > = 3.18.3 is installed, and then run the following command:

```bash
cd ~/mindquantum
bash build.sh --gitee
```

Here `--gitee` let the script download the third-party dependencies from the gitee code hosting platform. If you need to compile the GPU version, please first install CUDA 11.x and the corresponding graphics driver, and then use the `--gpu` parameters to execute the following compilation instructions:

```bash
cd ~/mindquantum
bash build.sh --gitee --gpu
```

Make **Windows system** sure that MinGW-W64 and CMake > = 3.18.3 are installed, and then run the following command:

```bash
cd ~/mindquantum
build.bat /Gitee
```

Make **Mac system** sure that OpenMP and CMake > = 3.18.3 are installed, and then run the following command:

```bash
cd ~/mindquantum
bash build.sh --gitee
```

- Install the compiled whl package

    Enter the output directory and install the compiled mindquantum whl package through the `pip` command.

#### Verify installation successful

Execute the following command. If no error `No module named 'mindquantum'` is reported, the installation is successful.

```bash
python -c 'import mindquantum'
```

### Compile

MindQuantum provides **Compile the package** and **Local compilation** two methods

1.Compile the package. If the user needs to adapt to different system environments and python versions, he can compile the source code into a package and then install it. For detailed process, please refer to the above *Source code installation*

2.Local compilation. If the user adds or modifies part of the code and needs to quickly verify whether it is effective, he can use local compilation to compile the C + + code into a *.so file, and then use Python to call and set the environment variables. There is no need to recompile the package, and then uninstall and reinstall it, which is convenient for debugging and verification.

Download the source code from the code warehouse

```bash
cd ~
git clone https://gitee.com/mindspore/mindquantum.git
```

- **Linux** System, depending on CMake > = 3.18.3, compiled natively.

   `--gitee` The parameter specifies that the script downloads third-party dependencies from the gitee code hosting platform; `export` the command adds the mindquantum source code path to the PYTHONPATH environment variable.

  ```bash
  cd ~/mindquantum

  bash build_locally.sh --gitee
  export PYTHONPATH=`pwd`$PYTHONPATH
  ```

- **Windows** system, depending on MinGW-W64 and CMake > = 3.18.3, compiled natively.

  ```bash
  cd ~/mindquantum
  build_locally.bat /Gitee -G 'MinGW Makefiles'
  set PYTHONPATH=%cd%;%PYTHONPATH%
  ```

- **Mac** System, depending on OpenMP and CMake > = 3.18.3, compiled natively.

  ```bash
  cd ~/mindquantum
  bash build_locally.sh --gitee
  export PYTHONPATH=`pwd`$PYTHONPATH
  ```

- System Dependent Component Installation
    - GCC-GNU Compiler Suite, Installation Instruction

      1.Linux and MacOS installation

      1.1 Linux/Ubuntu/Centos systems and MacOS systems will come with versions `gcc` that meet the MindQuantum compilation.

      1.2 Use the command to install GCC

      ```shell
        # Ubuntu install
        apt-get update
        apt-get install gcc
        apt-get install build-essential

        # Centos install
        yum install gcc gcc-c++

        # output gcc version,verify successful installation
        gcc --version
      ```

      2.Window installation

        2.1 Window's C/C + + compiler suite generally recommends Mingw-w64, which can be used to compile and run Window applications and DLL files.

        2.Enter the [Mingw-w64 installation package](https://sourceforge.net/projects/mingw-w64/files/) page and download the offline installation package. The recommended **x86_64-posix-seh** version here corresponds to the x64 architecture.

        2.3. Double-click locally to open the installation package and install it according to the prompt. (Note that the environment variable is written)

        2.4 After installation, open the cmd terminal and enter the command

      ```shell
      C:\Users\xx> gcc --version

      gcc (x86_64-win32-seh-rev0, Built by MinGW-W64 project) 8.1.0
      Copyright (C) 2018 Free Software Foundation, Inc.
      ```

      Encounter bug, can send issue, or Google Baidu, reference [csnd blog](https://blog.csdn.net/jiqiren_dasheng/article/details/103775488).

      CMake-Cross-platform open source builder, installation instructions

      1.Enter the [CMake](https://cmake.org/) official website and select the appropriate version of the installation package according to the system.

      Windows system recommend **cmake-3.2\*-windows-x86_64.msi** ,

      Macos system recommend **cmake-3.2\*-macos-universal.dmg** ,

      Linux system recommendation **cmake-3.2\*-linux-x86_64.sh**.

      2.Double-click locally to open the CMake installation package, install it according to the prompt, note that CMake needs to be added to the user variable, and finally click `Finish` to complete the installation.

        3.Open the terminal, output the command in the command line, and output the version number to indicate success.

      ```shell
        cmake --version
        >>> cmake version 3.24.2
      ```

### Development

The mindquantum is mainly developed in C + + and python, the core computing unit is implemented in C/C + +, and the upper interface and peripheral modules are implemented in Python

The development process is mainly divided into two categories, developing new functions and fixing bugs:

- Develop new features, enter the issue page of MindQuantum, submit the issue, write new feature descriptions, categories, implementation methods, etc. Talk to the MindQuantum development team to confirm if this feature is necessary. Start the implementation locally, write the code, implement the function, write the corresponding test cases, and the corresponding documentation. Submit PR, merge into the main branch after the code passes the review, and complete the development of new functions.

- Fix the bug, go to the issue page of MindQuantum, read the unclosed issues, and claim the issue to solve the problem. Or if you encounter a bug when using MindQuantum, you are welcome to submit an issue to help improve the MindQuantum function module.

## Quick start

- On [Gitee](https://gitee.com/mindspore/mindquantum) the fork mind quantum code bin.
- See [README.md](https://gitee.com/mindspore/mindquantum/blob/master/README.md) for project information and build instructions.
- First experience, Build a parameterized quantum circuit. Use `mindquantum` to a quantum circuit including a H-gate, an RX gate, and an ry gate, and obtain a quantum state

```python
from mindquantum import *
import numpy as np

encoder = Circuit().h(0).rx({'a0': 2}, 0).ry('a1', 1)
print(encoder)
print(encoder.get_qs(pr={'a0': np.pi / 2, 'a1': np.pi / 2}, ket=True))
```

Running the above code will output the quantum circuit and the final state.

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

## Code structure

- [`ccsrc`](https://api.gitee.com/mindspore/mindquantum/tree/master/ccsrc) core computing module, using C/C + + implementation
- [`cmake`](https://api.gitee.com/mindspore/mindquantum/tree/master/cmake) cmake compiles configuration information
- [`docs`](https://api.gitee.com/mindspore/mindquantum/tree/master/docs) Mind Quantum API Documentation
- [`mindquantum`](https://api.gitee.com/mindspore/mindquantum/tree/master/mindquantum) Mind Quantum quantum computing module, implemented in Python
    - [`mindquantum.dtype`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.dtype.html#module-mindquantum.dtype)  MindQuantum data type simulation.
    - [`mindquantum.core`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.html#module-mindquantum.core)  Core features of MindQuantum (eDSL).
        - [`gata`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/mindquantum.core.gates.html) quantum gate module providing different quantum gates.
        - [`circuit`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/mindquantum.core.circuit.html) Quantum circuit module, It can easily build quantum circuits that meet the requirements, including parameterized quantum circuits.
        - [`operators`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/mindquantum.core.operators.html) MindQuantum  Operator library
        - [`parameterresolver`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/mindquantum.core.parameterresolver.html)  The parameter parser module is used for declaring the used parameters.
    - [`mindquantum.simulator`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.simulator.html#module-mindquantum.simulator) A quantum simulator for simulating the evolution of a quantum system.
    - [`mindquantum.framework`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.framework.html#module-mindquantum.framework) Quantum neural network operators and cell.
    - [`mindquantum.algorithm`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.html#module-mindquantum.algorithm) Quantum algorithm.
        - [`compiler`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mindquantum.algorithm.compiler.html)  Quantum circuit compiling module
        - [`library`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mindquantum.algorithm.library.html) Common algorithm module
        - [`nisq`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mindquantum.algorithm.nisq.html) NISQ algorithm
        - [`error_mitigation`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mindquantum.algorithm.error_mitigation.html) Error mitigation module
        - [`mapping`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mindquantum.algorithm.mapping.html) Bit Mapping Module
    - [`mindquantum.device`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.device.html#module-mindquantum.device)  Mind Quantum hardware module.
    - [`mindquantum.io`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.io.html#module-mindquantum.io)  Input/output module for MindQuantum.
    - [`mindquantum.engine`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.engine.html#module-mindquantum.engine)  MindQuantum Engine Module.
    - [`mindquantum.utils`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.utils.html#module-mindquantum.utils) utility.
- [`mindquantum_config`](https://api.gitee.com/mindspore/mindquantum/tree/master/mindquantum_config) Item Configuration Information
- [`scripts`](https://api.gitee.com/mindspore/mindquantum/tree/master/scripts) Compiling a Dependent Tools Update Script
- [`tests`](https://api.gitee.com/mindspore/mindquantum/tree/master/tests) MindQuantum unit tests, based on Pytest, written in Python
- [`third_party`](https://api.gitee.com/mindspore/mindquantum/tree/master/third_party) Third-party open source packages on which MindQuantum compiles
- [`tutorials`](https://api.gitee.com/mindspore/mindquantum/tree/master/tutorials) Mind Quantum tutorial, which can be run directly with jupyter. Readable in [Official document of MindSpore](https://www.mindspore.cn/mindquantum/docs/en/master/index.html).

## Unit testing

MindQuantum writes unit test cases based on pytest. It is recommended that developers write corresponding unit test cases after implementing a new function or module to ensure normal functions

## Write document

Instructions for writing documentation, MindQuantum has two main types of documentation:

- User-oriented documents: These are the documents that users see on [Mind Spore Quantum website](https://www.mindspore.cn/mindquantum/docs/en/master/index.html) the Internet, including the tutorial plans for circuit construction, simulator, algorithm implementation, etc., which are helpful to quickly get started and apply MindQuantum, as well as to learn quantum computing algorithms.
- Documentation for developers: Documentation for developers is distributed across the code `MindQuanntum/docs` base. If you're interested in adding new developer documentation, read this page on the wiki to learn about best practices and write comments after you've written the code. The API is collated by extracting comments from the code.

## Contribution Workflow

### Code style

Please follow this style to make MindSpore easy to review, maintain and develop.

- Coding guidelines

  The *Python* coding style suggested by [Python PEP 8 Coding Style](https://gitee.com/link?target=https%3A%2F%2Fpep8.org%2F) and *C++* coding style suggested by [Google C++ Coding Guidelines](https://gitee.com/link?target=http%3A%2F%2Fgoogle.github.io%2Fstyleguide%2Fcppguide.html) are used in MindSpore community. The [CppLint](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fcpplint%2Fcpplint), [CppCheck](https://gitee.com/link?target=http%3A%2F%2Fcppcheck.sourceforge.net), [CMakeLint](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fcmake-lint%2Fcmake-lint), [CodeSpell](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fcodespell-project%2Fcodespell), [Lizard](https://gitee.com/link?target=http%3A%2F%2Fwww.lizard.ws), [ShellCheck](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fkoalaman%2Fshellcheck) and [PyLint](https://gitee.com/link?target=https%3A%2F%2Fpylint.org) are used to check the format of codes, installing these plugins in your IDE is recommended.

- Unittest guidelines

  The *Python* unittest style suggested by [pytest](https://gitee.com/link?target=http%3A%2F%2Fwww.pytest.org%2Fen%2Flatest%2F) and *C++* unittest style suggested by [Googletest Primer](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fgoogle%2Fgoogletest%2Fblob%2Fmaster%2Fdocs%2Fprimer.md) are used in MindSpore community. The design intent of a testcase should be reflected by its name of comment.

- Refactoring guidelines

  We encourage developers to refactor our code to eliminate the [code smell](https://gitee.com/link?target=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FCode_smell). All codes should conform to needs to the coding style and testing style, and refactoring codes are no exception. [Lizard](https://gitee.com/link?target=http%3A%2F%2Fwww.lizard.ws) threshold for nloc (lines of code without comments) is 100 and for cnc (cyclomatic complexity number) is 20, when you receive a *Lizard* warning, you have to refactor the code you want to merge.

- Document guidelines

  We use *MarkdownLint* to check the format of markdown documents. MindSpore CI modifies the following rules based on the default configuration.

    - MD007 (unordered list indentation): The **indent** parameter is set to **4**, indicating that all content in the unordered list needs to be indented using four spaces.
    - MD009 (spaces at the line end): The **br_spaces** parameter is set to **2**, indicating that there can be 0 or 2 spaces at the end of a line.
    - MD029 (sequence numbers of an ordered list): The **style** parameter is set to **ordered**, indicating that the sequence numbers of the ordered list are in ascending order.

  For details, please refer to [RULES](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmarkdownlint%2Fmarkdownlint%2Fblob%2Fmaster%2Fdocs%2FRULES.md).

### Fork-Pull development model

- Fork MindSpore repository

  Before submitting code to MindSpore project, please make sure that this project have been forked to your own repository. It means that there will be parallel development between MindSpore repository and your own repository, so be careful to avoid the inconsistency between them.

- Clone the remote repository

  If you want to download the code to the local machine, `git` is the best way:

  ```shell
  # For GitHub
  git clone https://github.com/{insert_your_forked_repo}/mindspore.git
  git remote add upstream https://github.com/mindspore-ai/mindspore.git
  # For Gitee
  git clone https://gitee.com/{insert_your_forked_repo}/mindspore.git
  git remote add upstream https://gitee.com/mindspore/mindspore.git
  ```

- Develop code locally

  To avoid inconsistency between multiple branches, checking out to a new branch is `SUGGESTED`:

  ```shell
  git checkout -b {new_branch_name} origin/master
  ```

  Taking the master branch as an example, MindSpore may create version branches and downstream development branches as needed, please fix bugs upstream first. Then you can change the code arbitrarily.

- Push the code to the remote repository

  After updating the code, you should push the update in the formal way:

  ```shell
  git add .
  git status # Check the update status
  git commit -m "Your commit title"
  git commit -s --amend #Add the concrete description of your commit
  git push origin {new_branch_name}
  ```

- Pull a request to MindSpore repository

  In the last step, your need to pull a compare request between your new branch and MindSpore `master` branch. After finishing the pull request, the Jenkins CI will be automatically set up for building test. Your pull request should be merged into the upstream master branch as soon as possible to reduce the risk of merging.

### Report issues

A great way to contribute to the project is to send a detailed report when you encounter an issue. We always appreciate a well-written, thorough bug report, and will thank you for it!

When reporting issues, refer to this format:

- What version of env (mindspore, os, python etc) are you using?
- Is this a BUG REPORT or FEATURE REQUEST?
- What kind of issue is, add the labels to highlight it on the issue dashboard.
- What happened?
- What you expected to happen?
- How to reproduce it?(as minimally and precisely as possible)
- Special notes for your reviewers?

**Issues advisory:**

- **If you find an unclosed issue, which is exactly what you are going to solve,** please put some comments on that issue to tell others you would be in charge of it.
- **If an issue is opened for a while,** it's recommended for contributors to precheck before working on solving that issue.
- **If you resolve an issue which is reported by yourself,** it's also required to let others know before closing that issue.
- **If you want the issue to be responded as quickly as possible,** please try to label it, you can find kinds of labels on [Label List](https://gitee.com/mindspore/community/blob/master/sigs/dx/docs/labels.md)

### Propose PRs

- Raise your idea as an *issue* on [GitHub](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-ai%2Fmindspore%2Fissues) or [Gitee](https://gitee.com/mindspore/mindspore/issues)
- If it is a new feature that needs lots of design details, a design proposal should also be submitted.
- After reaching consensus in the issue discussions and design proposal reviews, complete the development on the forked repo and submit a PR.
- None of PRs is not permitted until it receives **2+ LGTM** from approvers. Please NOTICE that approver is NOT allowed to add *LGTM* on his own PR.
- After PR is sufficiently discussed, it will get merged, abandoned or rejected depending on the outcome of the discussion.

**PRs advisory:**

- Any irrelevant changes should be avoided.
- Make sure your commit history being ordered.
- Always keep your branch up with the master branch.
- For bug-fix PRs, make sure all related issues being linked.

### Local code self-test

In the development process, it is recommended to use the pre-push function to perform local code self-check. Code scanning similar to the Code Check stage of CI access control can be performed locally to improve the success rate of access control. For usage, see [Pre-push Quick Guide](scripts/pre_commit/README.md).

After the development is completed, suggest *vscode* or *pycharm* format functions to standardize the code style.
