# 安装

该文件旨在帮助您构建和安装 MindQuantum。

## 快速开始

### 构建二进制 Python wheel

为了生成 MindQuantum 的二进制 wheel，我们建议您使用 Pypa 中的 `build`：

```bash
cd mindquantum
python3 -m build .
```

然而这产生的二进制wheel，它可能依赖于您系统某处的一些外部库。因此，如果其他用户没有安装相同的库或装在不同的非标准位置，分发它们可能会遇到一些问题。

#### Delocating the wheels

为了确保所有必需的外部库依赖都包含在二进制wheel文件中，您可以指示二进制wheel构建时 _delocate_ 产生的wheel。在实践中，这意味着通过将它们直接集成到二进制wheel中，以删除对系统上外部（共享）库的任何依赖。

##### 使用 cibuildwheel

这是构建二进制wheel的首选方法，因为它依赖于Docker镜像（仅在Linux上）或MacOS和Windows上的标准Python发行版。

```bash
cd mindquantum
python3 -m cibuildwheel .
```

如果 `cibuildwheel` 无法自动检测您的平台，或者例如你想要在MacOS上构建Linux wheel，则可能需要指定平台：

```bash
cd mindquantum
python3 -m cibuildwheel --platform linux .
```

有关提供哪些平台的更多信息，请参见 `python3 -m cibuildwheel --help`。

###### Linux

在Linux上，您可以直接在任何机器上运行脚本，因为它使用Docker镜像来构建Delocated Binary Wheel。如果您想构建Linux二进制wheel，在MacOS或Windows上也是如此。

###### MacOS

在MacOS上，`cibuildwheel` 将在构建二进制wheel之前将官方Python发行版安装在您的系统上。除非您了解自己在做什么，否则不要在开发机上运行脚本。

###### Windows

在Windows上，`cibuildwheel` 将在构建二进制wheel之前使用系统上的NuGet安装官方Python发行版。除非您了解自己在做什么，否则不要在开发机上运行脚本。

##### 在你的本地设备上

如果您不想依靠 `cibuildwheel` （例如在MacOS上），也可以自动调用 `auditwheel` 或 `delocate`，通过指定 `MQ_DELOCATE_WHEEL` 环境变量来构建wheel之后。例如这样：

```bash
cd mindquantum
MQ_DELOCATE_WHEEL=1 python3 -m build .
```

如果您打算将wheel分发给可能与您拥有不同系统的人，我们强烈建议您尝试指定 `MQ_DELOCATE_WHEEL_PLAT` 环境变量。默认情况下，setup脚本在64位计算机上假设 `'Linux_x86_64'`，但您可以指定[auditwheel](https://github.com/pypa/auditwheel)支持的任何平台。为了将wheel分发给更多的受众，我们推荐设置 `MQ_DELOCATE_WHEEL_PLAT=manylinux2010_x86_64`，尽管如果您的编译器版本太新，则在delocate wheel时可能会导致错误。

### 本地构建 MindQuantum

您可以通过使用本地构建脚本之一来为本地开发设置MindQuantum：

- `build_locally.bat` (MS-DOS BATCH script)
- `build_locally.ps1` (PowerShell script)
- `build_locally.sh` (Bash script)

除了一些小差异，三个脚本的功能都是相同的。所有脚本都接受标志以显示帮助消息（`-h`， `--help`，`-H`，`-Help` 对于 Bash，PowerShell和MS-DOS batch）。请调用您选择的脚本，以查看其提供的最新功能集。

1. 设置Python虚拟环境；
2. 更新虚拟环境的软件包并安装一些必需的依赖；
3. 在Python虚拟环境中添加PTH文件，以确保MindQuantum能检测到；
4. 创建一个 `build` 目录并在其中运行 CMake；
5. 原地编译 MindQuantum。

下次运行脚本时，除非指定清洁选项之一或强制CMAKE配置步骤，否则该脚本只会重新编译Mindquantum。

有关更多信息，请查看脚本的帮助消息，您可以使用 `./build_locally.sh -h` 或 `./build_locally.sh --help`。输出如下所示，供参考。

## CMake 配置

### CMake 选项

这是所有可用于自定义的CMake选项的详尽列表：

| Option name                     | Description                                                           | Default value       |
|---------------------------------|-----------------------------------------------------------------------|---------------------|
| BUILD_SHARED_LIBS               | 构建共享的libs                                                          | OFF                 |
| BUILD_TESTING                   | 启用构建测试套件                                                         | OFF                 |
| CLEAN_3RDPARTY_INSTALL_DIR      | 清除第三方安装目录                                                       | OFF                 |
| CUDA_ALLOW_UNSUPPORTED_COMPILER | 允许使用CUDA不支持的编译器版本                                             | OFF                 |
| CUDA_STATIC                     | 使用 Nvidia CUDA 库的静态版本                                             | OFF                 |
| DISABLE_FORTRAN_COMPILER        | 对于一些第三方库，强制禁止 Fortran 编译器                                    | ON                  |
| ENABLE_CMAKE_DEBUG              | 启用详细输出来调试CMAKE                                                   | OFF                 |
| ENABLE_CUDA                     | 启用使用CUDA代码                                                           | OFF                 |
| ENABLE_GITEE                    | 使用gitee代替github作为（某些）第三方依赖                                   | OFF                 |
| ENABLE_MD                       | 编译时使用 /MD, /MDd 标志 (仅MSVC)                        | OFF                 |
| ENABLE_MT                       | 编译时使用 /MT, /MTd 标志 (仅MSVC)                        | OFF                 |
| ENABLE_PROFILING                | 启用编译分析标志                               | OFF                 |
| ENABLE_RUNPATH                  | link时优先使用 RUNPATH 而不是 RPATH                               | ON                  |
| ENABLE_STACK_PROTECTION         | 启用 编译期栈保护                            | ON                  |
| IN_PLACE_BUILD                  | 原地构建 C++ MindQuantum 库                         | OFF                 |
| IS_PYTHON_BUILD                 | 是否 CMake 被 setup.py 调用                                | OFF                 |
| LINKER_DTAGS                    | link期间启用 --enable-new-dtags (或者使用 --disable-new-dtags)  | ON                  |
| LINKER_NOEXECSTACK              | link期间使用 `-z,noexecstack` (如果支持的话)                    | ON                  |
| LINKER_RELRO                    | link期间使用 `-z,relro` (如果支持的话)                          | ON                  |
| LINKER_RPATH                    | 编译期间使用 RUNPATH/RPATH 相关的标志                       | ON                  |
| LINKER_STRIP_ALL                | link期间使用 `--strip-all` (如果支持的话)                       | ON                  |
| USE_OPENMP                      | 使用 OpenMP 用于并行计算                            | ON                  |
| USE_PARALLEL_STL                | 使用 parallel STL 用于并行计算 (使用 TBB 或者别的)          | OFF                 |
| USE_VERBOSE_MAKEFILE            | 生成详细 Makefiles (如果支持的话)                             | ON                  |

下面是以上一些选项的更明确的描述：

#### `CLEAN_3RDPARTY_INSTALL_DIR`

这将删除本地安装目录中的任何预先存在的安装（默认情况下 `/path/to/build/.mqlibs`） _除了_ 当前基于第三方库的哈希当前所需的安装。

#### `DISABLE_FORTRAN_COMPILER`

目前这仅在安装 Eigen3 时才有效果。

### CMake 变量

除上述CMAKE选项外，您还可以通过某些特殊的CMAKE变量来自定义构建。这些细节在下面描述。

#### `MQ_FORCE_LOCAL_PKGS`

该变量的值对大小写不敏感。它可能是：

- 一个字符串（`all`）
- 一个或多个Mindquantum的第三方依赖的CMAKE软件包名称，被逗号分隔的列表（例如 `gmp,eigen3`）

列出的所有软件包将在CMAKE配置过程中本地编译。

#### `MQ_XXX_FORCE_LOCAL`

将其设置为Mindquantum的第三方依赖之一，将导致在CMake配置过程中本地编译这些软件包。请注意，软件包名称 `XXX` 必须全部是大写字母。
