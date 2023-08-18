import sys

sys.path.append("./src")

from src.main import Main
import numpy as np
from src.main import*
origin_test_data = np.load("src/train.npy", allow_pickle=True)[0]

              # 通过NumpySlicesDataset创建测试样本的数据集，batch(5)表示测试集每批次样本点有5个





origin_test_x = origin_test_data['train_x']
origin_test_y = origin_test_data['train_y']
#origin_test_y = origin_test_y.astype(np.int32)
#origin_test_y=  origin_test_y.reshape(1,origin_test_y.shape[0])
main = Main()
main.train()
main.export_trained_parameters()
#main.load_trained_parameters()

#predict = np.argmax(ops.Softmax()(main.model.predict(Tensor(X_test))), axis=1)    # 使用建立的模型和测试样本，得到测试样本预测的分类
#correct = model.eval(test_loader, dataset_sink_mode=False)                   # 计算测试样本应用训练好的模型的预测准确率
#print(correct)
#main.test()
#predict = main.predict(origin_test_x)
#acc = np.mean(predict == origin_test_y)
#print(f"Acc: {acc}")
#main.load_trained_parameters()
#predict = main.predict(X_test)
#acc = np.mean(predict == y_test)
#print(f"Acc: {acc}")
predict = main.predict(origin_test_x)
print(predict.shape)
print(origin_test_y.shape)
acc = np.mean(predict == origin_test_y)
print(f"Acc: {acc}")