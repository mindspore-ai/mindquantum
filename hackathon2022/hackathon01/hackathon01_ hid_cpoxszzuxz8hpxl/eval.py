import sys

sys.path.append("./src")

from src.main import Main
import numpy as np

origin_test_data = np.load("src/train.npy", allow_pickle=True)[0]
origin_test_x = origin_test_data['train_x']
origin_test_y = origin_test_data['train_y']
# origin_test_x = origin_test_x[:500,:]
# origin_test_y = origin_test_y[:500]
main = Main()
predict = main.predict(origin_test_x)
acc = np.mean(predict == origin_test_y )
print(f"Acc: {acc}")