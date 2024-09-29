# MindQuantum GPU算子封装指南

本文直白并具体地展示了一个 MindQuantum GPU 算子的相关文件和封装流程，旨在帮助不熟悉 MindQuantum GPU 算子开发和封装流程的开发者了解细节并能参考此流程模板快速开发能够集成到 MindQuantum 框架的 GPU 算子。

------

对于一个 mindquantum GPU 算子的开发，主要包括以下三个部分：

1. 头文件定义
2. 算子实现
3. python 模块绑定

以 algorithm/qaia/sb 的 GPU 算子（朴素版）开发为例，调用了 cuBLAS 库。以下是相关代码文件列表和注意事项。

## 相关代码文件列表



```
.
├── ccsrc  
│   ├── include
│   │   ├── algorithm
│   │   │   └── qaia
│   │   │       └── detail
│   │   │           └── gpu_sb.cuh // 头文件定义
│   │   └── CMakeLists.txt // 在install(DIRECTORY)中添加 algorithm/
|   ├── lib
│   │   ├── algorithm
│   │   │   ├── qaia
│   │   │   │   └── detail
│   │   │   │       ├── gpu_sb_update.cu // GPU 算子的具体实现
│   │   │   │       └── CMakeLists.txt // 将当前目录下的 .cu 文件加入 target_sources
│   │   │   └── CMakeLists.txt // 注意事项见下文
│   │   └── CMakeLists.txt // 添加 add_subdirectory(algorithm)
|   └── python
│       └── algorithm
│           ├── include
│           ├── lib
│           │   └── _qaia_sb.cu // 使用 pybind11 定义模块 _qaia_sb
│           └── CMakeLists.txt // pybind11_add_module 导入模块 _qaia_sb
├── mindquantum 
|   └── algorithm
|       └── qaia
|           └── SB.py // 导入自定义模块并调用算子
└── setup.py // 算子注册
```

## 注意事项

关于 **ccsrc/lib/algorithm/CMakeLists.txt** 

若使用 CUDA 库函数，要在 target_link_libraries 加入其名称。例如 ${CUDA_HOME}/lib64 中的 libcublas.so 对应 cublas，以此类推。见下图 CMakeLists.txt 示例：

![](figures\cmake_lib.png)



关于 **setup.py**

在 ext_modules 中加入自定义模块，见下图：

<img src="figures\setuppy.png" style="zoom:60%;" />