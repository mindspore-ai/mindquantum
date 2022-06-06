# 2022黑客松赛题-量子化学模拟:基态能求解的浅层线路设计

# 代码大致介绍
在’VQEoptimizer‘ class里面我们基于VQE算法完成了分子基态能求解的流程。其中’remove_orbital‘函数由约化密度矩阵分析，对哈密顿量做Mean Field Approximation可以去除
一些不需要的spin orbital；’generate_circuit‘函数根据参数实现了简化版的UCCSD ansatz和我们设计的Pair-Excitation circuit；’optimize‘函数调用mindspore里面的
’Adagrad‘优化器完成变分过程；’imag_time_evolution‘函数大致实现了变分虚时演化算法。

# 联系方式
如果您对我们的code感兴趣的话，欢迎联系我进一步深入探讨:-）
我的email：CHU_YI_DA@163.com

If you are interesting in our project，feel free to contact me:-)
My email：CHU_YI_DA@163.com