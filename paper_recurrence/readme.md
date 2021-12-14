# 论文复现大赛

- [论文复现大赛](#论文复现大赛)
  - [量子计算论文复现大赛开发指南](#量子计算论文复现大赛开发指南)
  - [可选论文](#可选论文)
  - [代码要求](#代码要求)
  - [代码提交格式](#代码提交格式)
  - [代码提交路径](#代码提交路径)
  - [评分标准](#评分标准)
    - [复现结果审查阶段](#复现结果审查阶段)
    - [入围决赛评选阶段](#入围决赛评选阶段)

## 量子计算论文复现大赛开发指南

[量子计算论文复现大赛开发指南](https://gitee.com/mindspore/mindquantum/tree/research/paper_recurrence/developers_guide)

## 可选论文

[https://competition.huaweicloud.com/information/1000041627/circumstance](https://competition.huaweicloud.com/information/1000041627/circumstance)

## 代码要求

请使用[**MindQuantum**](https://gitee.com/mindspore/mindquantum)量子计算库和[**MindSpore**](https://www.mindspore.cn/install)机器学习框架（如有需求）来复现您选择的论文，如有部分功能MindQuantum中尚未实现，请使用Numpy或者Scipy科学计算包来实现。

## 代码提交格式

请参考[**paperid_username_for_example**](https://gitee.com/mindspore/mindquantum/tree/research/paper_recurrence/paperid_username_for_example)来组织和提交您的代码，主要包含如下三大块：`src`、`main.ipynb`和`readme.md`。`src`存放您论文复现的代码，`main.ipynb`是用来介绍和展示您复现结果的jupyter notebook文件，`readme.md`是用来对项目进行介绍的简要文档。

## 代码提交路径

- 初赛作品提交：将代码和文档按照**paperid_username_for_example**格式要求提交到mindquantum仓research分支的**paper_recurrence/paperid_username**。**paperid**为论文的序号，**username**为参赛者华为云账号名。

## 评分标准

分赛事分为两个阶段：1、复现结果审查阶段；2、入围决赛评选阶段；

### 复现结果审查阶段

只需达到相应论文的**复现最低要求**即可完成复现结果审查阶段。

### 入围决赛评选阶段

在入围决赛评选阶段，评委会根据您复现的结果在如下五个维度进行打分：

|评分维度|解释|
|-|-|
|复现完成度|在最低要求的基础之上，是否复现更多的论文内容|
|复现精度|复现模型在你的调教之下，是否比原论文中精度更高，如已达到100%等特殊情况，本单项直接获得满分|
|代码质量|根据代码符合编程规范的程度来评判，可参考MindQuantum的源代码编码规范|
|性能|根据代码复现过程中CPU占用量、内存占用量和计算时长等来判定|
|创新性|在原论文的基础上，是否有更多的自己的思考，是否对模型有优化等|

> 评选规则解释权归本次大赛组委会所有