- 运行如下命令，并将命令中的`your_name`换为你的名字。此命令会复制课堂任务文件，并以你的名字结尾重命名，此举避免提交PR时冲突

```bash
./copy.sh your_name
```

- 本次讲解内容是基于MindQuantum的dev分支的最新特性，因此需要安装dev分支的whl包，请执行如下命令

```bash
pip install mindquantum-0.3.0-py3-none-any.whl
```

- 更新cmake，如有需要
```bash
cd ~
git clone https://gitee.com/donghufeng/cmake_mirror.git
cd cmake_mirror
./install.sh
source ~/.bashrc
```