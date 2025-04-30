# MindQuantum 案例

本目录收集了使用MindQuantum框架实现的各种算法案例，旨在帮助开发者更好地理解和使用MindQuantum。

## 案例列表

- [Q-JEPG算法](./quantum-jpeg/quantum-jpeg.ipynb)
- [基于 MindSpore Quantum 实现 Adapt Clifford 算法求解 Maxcut 问题](./Adapt-Clifford/Adapt-Clifford-for-MaxCut.ipynb)
- [深度展开鲁棒量子启发式算法](./quantum-dusb/DU_SB_test.ipynb)
- [基于 DQAS 算法的 MNIST 手写数字二分类问题](./DQAS_for_image_classfication/DQAS_for_image_classfication.ipynb)

## 贡献指南

欢迎贡献新的案例！提交案例时请注意以下几点：

1. 推荐以一个独立的文件夹组织案例，结构示例如下：

    ```text
    your_case_name/
    ├── main.ipynb          # 主逻辑与核心流程展示（必须能够完整运行）
    ├── utils/              # 辅助函数或模块（可选）
    │   ├── helper.py
    │   └── preprocess.py
    └── README.md           # 案例简介与数据获取方式说明（可选）
    ```

    - `main.ipynb` 中应清晰展示案例的核心逻辑和关键步骤，配合详细的文字说明和注释。
    - 辅助函数、数据处理、模型定义等细节代码可放置于单独的 Python 文件中，并在 notebook 中调用，以保持 notebook 的简洁性和可读性。
    - 请确保提交的 notebook 能够完整运行（从头到尾执行所有单元格），避免出现运行错误或缺失依赖。

2. 每张图片大小不超过 100KB。请勿上传数据集或其他大文件，在文档中注明数据获取方式即可<sup>1</sup>。

3. 确保所有代码符合 PEP8 规范。推荐使用 IDE（如 PyCharm、VS Code 等）的自动格式化功能，或使用 autopep8、black 等工具进行代码格式化。

4. 提交 PR 后，在评论区发送 `/retest` 触发门禁，根据代码检查结果修改代码，直到通过所有代码检查。

> [1] 难以从公共渠道获取的大文件可以传到自己的免费[华为云OBS桶](https://support.huaweicloud.com/obs/index.html)里，创建分享链接，在文中标注通过链接获取。

## 使用 Black 自动格式化代码

建议使用 Black 工具自动格式化 Python 代码和 Jupyter Notebook 文件，以确保代码风格统一，减少或避免代码提交时门禁检查出现的格式错误。

### 安装 Black（支持 Jupyter Notebook）

```bash
pip install "black[jupyter]"
```

### 使用方法

- **格式化单个 Jupyter Notebook 文件：**

```bash
black your_notebook.ipynb
```

- **格式化指定目录下的所有 Python 和 Jupyter 文件：**

```bash
black <your_code_directory>
```

- **检查哪些文件需要格式化（仅检查，不实际修改文件）：**

```bash
black --check <your_code_directory>
```

## 环境要求

- Python >= 3.7
- MindQuantum >= 0.9.11

## 参考资源

- [MindQuantum 官方文档](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/index.html)


