# AI Gallery分享操作指导


1. 点击此处[https://console.huaweicloud.com/modelarts/?region=cn-north-4#/dev-container/create](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/dev-container/create)，在ModelArts控制台创建notebook开发环境。注意领取代金券，选择合适规格避免欠费。

2. 进入notebook开发环境，创建对应Kernel的notebook，完成论文复现。可参考小样本学习[https://developer.huaweicloud.com/develop/aigallery/notebook/detail?id=3a1b5b56-abd8-449e-839d-b3adf48c6599](https://developer.huaweicloud.com/develop/aigallery/notebook/detail?id=3a1b5b56-abd8-449e-839d-b3adf48c6599)

说明：ModelArts进去的Jupyter Notebook里的MindQuantum镜像需要升级到0.5.0。

```bash
pip install https://hiq.huaweicloud.com/download/mindquantum/newest/linux/mindquantum-master-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. 代码、数据和依赖文件可以打包上传至指定OBS文件目录

```bash
import  moxing as mox    # 华为自研库
mox.file.copy_parallel(‘xxx’, ’ obs://obs-aigallery-developer/MindQuantum/xxx’) # xxx 命名与gitee代码仓保持一致 ，如 YOLO_SPY，尽量不要重名
```

4. 论文复现的notebook第一步，可以下载上述文件，并进入该文件目录

```bash
# 获取代码和数据
import moxing as mox
mox.file.copy_parallel('obs://obs-aigallery-developer/MindQuantum/xxx','xxx')
%cd xxx
```

如果有依赖库，需要在线安装，可使用pip install方式

接下来进行训练，推理即可

5. 点击右上角分享按钮，发布到AI Gallery（分享主文件）

6. 发布成功之后，可以修改副标题和页面，确认一键run in ModelArts可以成功运行代码，即可领取奖品