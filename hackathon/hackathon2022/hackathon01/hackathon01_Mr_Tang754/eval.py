import sys

#sys.path.append("./src")
sys.path.append('/home/user/QCNN/QCNN/src') # 加入mypackge的父模块

from src.main import Main
import numpy as np

#origin_test_data = np.load("test.npy", allow_pickle=True)[0]
origin_test_data = np.load("/home/user/QCNN/QCNN/train.npy", allow_pickle=True)[0]

# origin_test_x = origin_test_data['test_x']
# origin_test_y = origin_test_data['test_y']

origin_test_x = origin_test_data['train_x']
origin_test_y = origin_test_data['train_y']

main = Main()
main.load_trained_parameters()
predict = main.predict(origin_test_x)
acc = np.mean(predict == origin_test_y)

print(f"Acc: {acc}")

print(origin_test_y.shape)
print(predict.shape)
print(origin_test_y)
print(predict)
