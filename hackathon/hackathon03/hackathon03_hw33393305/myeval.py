from main import Main
import numpy as np

origin_test_x = np.load("test_x.npy", allow_pickle=True)
# origin_test_y = np.load("train_y.npy", allow_pickle=True)
# origin_test_data = np.load("mytest.npy", allow_pickle=True)[0]
# origin_test_x = origin_test_data['test_x']
# origin_test_y = origin_test_data['test_y']

main = Main()
main.train()
main.export_trained_parameters()
main.load_trained_parameters()
predict = main.predict(origin_test_x)
np.save('predict_y.npy', predict)
