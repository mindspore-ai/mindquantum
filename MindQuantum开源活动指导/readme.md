# MindQuantum开源活动指导

- [MindQuantum开源活动指导](#mindquantum开源活动指导)
  - [准备阶段](#准备阶段)
    - [登录HiQ量子计算云服务](#登录hiq量子计算云服务)
	- [CloudIDE使用指南](#cloudide使用指南)
    - [Watch & Star & Fork代码仓](#watch--star--fork代码仓)
    - [导入MindQuantum的代码仓](#导入mindquantum的代码仓)
    - [开发示例](#开发示例)
  - [参与开源互动热身](#参与开源互动热身)
    - [参与开源热身（参与线上会议实操互动）](#参与开源热身参与线上会议实操互动)
    - [PR贡献奖（会议中/后）](#pr贡献奖会议中后)
    - [需要大家邮件反馈如下内容](#需要大家邮件反馈如下内容)
  - [部分奖品展示](#部分奖品展示)

## 准备阶段

![](./images/1.png)

基于HiQ量子计算云服务和MindQuantum开源量子计算框架开发算法。

### 登录HiQ量子计算云服务

点击HiQ量子计算云服务[https://hiq.huaweicloud.com/portal/home](https://hiq.huaweicloud.com/portal/home)链接（进入 HiQ 官网[https://hiq.huaweicloud.com/](https://hiq.huaweicloud.com/) ，点击右上角【新版】按钮）跳转至登录入口（华为云账号登录，若没有华为云账号，请先注册华为云账号并实名认证）

![](https://hiq.huaweicloud.com/hiq_assets/hiq_news_images/doc/1.png)

![](https://hiq.huaweicloud.com/hiq_assets/hiq_news_images/doc/2.png)

### CloudIDE使用指南

1. 创建 HiQ 实例

点击【新建实例】按钮，进入实例参数配置界面。

![](https://hiq.huaweicloud.com/hiq_assets/hiq_news_images/doc/3.png)

2. 基础配置

实例名称：默认随机分配，可点击修改，名称以字母数字开头和结尾，长度介于 3~20 个字符。
描述：不能包含&、<>、/、’、”字符，长度介于 0~100 之间。
基础配置：使用 x86CPU 架构、2U8G、5GB 存储容量。
自动休眠：默认自动休眠时长为 1 小时，可选择 1 小时和 24 小时（若实例长时间无操作，将自动休眠），
请点击【下一步】按钮，进入工程配置界面。

![](https://hiq.huaweicloud.com/hiq_assets/hiq_news_images/doc/4.png)

3. 工程配置

来源：默认选择样例工程，CloudIDE 内置 MindQuantum、HiQSimulator 等多种样例代码供学习使用。
名称：默认随机分配，可点击修改，需输入数字或字母，长度介于 3~20 字符。
请点击【确定】按钮，进入编程界面。

![](https://hiq.huaweicloud.com/hiq_assets/hiq_news_images/doc/5.png)

4. 体验编程

（1）新建文件
方法一：点击左上角  ，选择 文件 > 新建文件，输入文件名称.py，点击【确定】按钮。

![](https://hiq.huaweicloud.com/hiq_assets/hiq_news_images/doc/6.png)

方法二：点击上方 ，输入文件名称.py，点击【确定】按钮。

![](https://hiq.huaweicloud.com/hiq_assets/hiq_news_images/doc/7.png)

（2）运行文件

在插件中运行的任务可以保存和查看，操作步骤如下：

打开终端 Terminal

方法一：使用Ctrl + `快捷键。

方法二：点击上方  ，新建终端。

![](https://hiq.huaweicloud.com/hiq_assets/hiq_news_images/doc/8.png)

保存结果

选择需要需要运行文件，右键 > 在终端中打开，输入命令python3 文件名 >> 结果文件名，回车，运行后的结果将会保存在结果文件里，可在左侧资源管理器查看。

![](https://hiq.huaweicloud.com/hiq_assets/hiq_news_images/doc/9.png)

![](https://hiq.huaweicloud.com/hiq_assets/hiq_news_images/doc/10.png)


### Watch & Star & Fork代码仓

1. 注册并登录Gitee，访问MindQuantum主仓库[https://gitee.com/mindspore/mindquantum](https://gitee.com/mindspore/mindquantum)
2. Watch并Star和Fork MindQuantum的主仓到个人空间。（已Forked可忽略）

![](./images/12.png)

3. Watch并Star和Fork MindSpore的主仓到个人空间。（已Forked可忽略）[https://gitee.com/mindspore/mindspore](https://gitee.com/mindspore/mindspore)

### 导入MindQuantum的代码仓

1. 获取个人空间Forked的代码仓链接地址：

![](./images/13.png)
![](./images/14.png)
![](./images/15.png)

2. 进入创建的CloudIDE实例，在菜单中选择“文件/导入项目”

![](./images/16.png)

3. 弹出“导入项目”的窗口。填写已Forked的MindQuantum的 URl、Gitee的账号和密码。

![](./images/17.png)

4. 导入成功后，选择“打开项目”

![](./images/18.png)

5. 在CloudIDE左下角 点击master，切换到research分支。

![](./images/19.png)

6. 将如下命令 复制粘贴到CloudIDE的Terminal终端里面，安装最新版本mindquantum。

```bash
pip install https://hiq.huaweicloud.com/download/mindquantum/newest/linux/mindquantum-master-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

7. 接下来在终端设置提交代码时附带的提交信息，包括用户名和邮箱，注意需要跟gitee个人信息页面上的一致。

![](./images/20.png)

### 开发示例

1. 新建一个用于测试的Jupyter Notebook文件，并在Notebook中尝试调用MindQuantum。为了显示效果，可通过Ctrl+Shift+P调出CloudIDE的控制面板，并在其中输入color theme，选择一个亮色调的主题。后面可以尝试使用MindQuantum的量子线路测量模块。

![](./images/21.png)

2. 完成开发后，请点击保存按钮来保存Notebook的修改，并通过CloudIDE提交修改到远程分支。

![](./images/22.png)

3. 点击面板左边的源代码控制标签页，在点击加号，将需要修改的文件暂存起来，在上方输入框内填写提交信息。

![](./images/23.png)

4. 点击左下角的同步按钮，将CloudIDE中的更改提交到自己的远程仓库中。远程仓库也能看到相应的更新。
5. 将自己仓库的更新通过pull request的方式提交到mindquantum的主仓库，完成最终的代码提交。

![](./images/24.png)

6. 这里将源分支和目标分支选为mindspore/mindquantum的research分支。填写PR标题, 取消勾选【合并后关闭提到的issue】。选择所需的审查人员后，即可创建PR。

![](./images/25.png)

7. 签署CLA。对于第一次参与MindQuantum开源开发的同学，在评论区会发现没有签署CLA。请进入签署页面，选择sign individual cla，并根据gitee上的个人信息，填写签署信息，完成CLA的签署。回到PR页面，在评论区回复 /check-cla，检查cla是否签署完成，如果没有，则需稍等片刻。签署完后，提交PR过程结束。

![](./images/26.png)

- MindQuantum开源活动指导视频

[https://www.bilibili.com/video/BV1mu411d7ET](https://www.bilibili.com/video/BV1mu411d7ET)

## 参与开源互动热身

### 参与开源热身（参与线上会议实操互动）

### PR贡献奖（会议中/后）

欢迎大家在会上跟专家实操互动。根据提交PR结果，前3名会奖励**高级定制背包**；4~10名会奖励**定制马克杯**; 其他同学提交PR审核通过就会奖励**布袋/书/雨伞**等奖品随机发送。（活动详情会议中会讲解，以上奖品最终以实际库存为准。）
> 注意：大家提交PR后一定要按照下面的要求发邮件反馈基本信息到公共邮箱，才能拿到奖品哦！！！

### 需要大家邮件反馈如下内容

1. 主送邮箱：hiqinfo1@huawei.com
2. 邮件反馈内容

| 主题                | \*\*月\*\*日MindQuantum开源活动体验                                  |
| ------------------- | -------------------------------------------------------------------- |
| PR地址              | https://gitee.com/mindspore/mindquantum/pulls/XXX 下面有详细路径截图 |
| 邮寄地址            |                                                                      |
| 收货人姓名+手机号码 |                                                                      |

上述信息仅用于邮寄奖品，不做其他用途。
PR地址：
![](./images/27.png)

## 部分奖品展示

![](./images/28.png)

欢迎点击了解更多MindQuantum知识！

MindQuantum官网：[https://www.mindspore.cn/mindquantum](https://www.mindspore.cn/mindquantum)

Gitee代码仓：[https://gitee.com/mindspore/mindquantum](https://gitee.com/mindspore/mindquantum)

**期待您成为新时代的开源社区贡献者，加入MindQuantum的开发者行列，共同携手推进量子计算的发展！**

![](./images/29.png)

量子计算小助手微信