# MindQuantum 构建二进制whl

如果你想构建MindQuantum的二进制whl，然后将它们重新分配给其他人，你可能想考虑将whl定位，以确保没有与你的特定系统相关的外部依赖。

目前，只有Linux和MacOS被支持。

## 构建whl

为了生成MindQuantum的二进制whl，我们推荐你使用Pypa的`build`包。

```bash
cd mindquantum
python3 -m build .
```

然而，这将产生二进制whl，它可能依赖于你系统中某个地方的一些外部库。因此，如果其他用户没有安装相同的库，或者没有安装在不同的非标准位置，分发这些库可能会遇到一些问题。

## ＃＃定位whl

为了确保所有需要的外部库依赖都包含在二进制whl文件中，你可以指示二进制whl的构建过程来_delocate_产生的whl。在实践中，这意味着通过将外部（共享）库直接集成到二进制whl中来消除对你系统中的任何依赖。

### 使用cibuildwheel

这是构建二进制whl的首选方式，因为它依赖于Docker镜像（仅在Linux上）或MacOS和Windows上的标准Python发行版。

```bash
cd mindquantum
python3 -m cibuildwheel .
```

如果`cibuildwheel`不能自动检测你的平台，或者你想在MacOS上构建Linux wheel，你可能需要指定你的平台。

```bash
cd mindquantum
python3 -m cibuildwheel --platform linux .
```

参见`python3 -m cibuildwheel --help`以了解更多关于哪些平台可用的信息。

#### Linux

在Linux上，你可以在任何机器上直接运行脚本，因为它是使用Docker镜像来构建脱机的二进制whl。如果你想构建Linux二进制whl，在MacOS或Windows上也是一样的。

#### MacOS

在MacOS上，cibuildwheel会在构建二进制whl之前在你的系统上安装官方Python发行版。这使得在你的开发机器上运行该脚本并不合适，除非你明白自己在做什么。

#### Windows

在 Windows 上，cibuildwheel 将在构建二进制whl之前使用 NuGet 在你的系统上安装官方 Python 发行版。这使得在你的开发机器上运行该脚本不合适，除非你明白你在做什么。

### 在你的本地机器上

如果你不想依赖`cibuildwheel`机器(例如在MacOS上)，你也可以通过指定`MQ_DELOCATE_WHEEL`环境变量在构建whl后自动调用`auditwheel`或`delocate`，像这样。

```bash
cd mindquantum
MQ_DELOCATE_WHEEL=1 python3 -m build .
```

如果你打算把whl分发给其他可能与你的系统不一样的人，我们强烈建议你尝试指定`MQ_DELOCATE_WHEEL_PLAT`环境变量。默认情况下，设置脚本假设64位机器上的'linux_x86_64'，但你可以指定[auditwheel](https://github.com/pypa/auditwheel)支持的任何平台。为了将你的whl分发给更多的人，我们建议设置`MQ_DELOCATE_WHEEL_PLAT=manylinux2010_x86_64`，尽管如果你的编译器的版本太新，这可能会导致whl的定位错误。
