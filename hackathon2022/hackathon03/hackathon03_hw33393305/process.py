from myencoder import generate_value
import numpy as np

origin_train_x = np.load("train_x.npy", allow_pickle=True)
origin_train_y = np.load("train_y.npy", allow_pickle=True)

encoder_value = list()
for i in range(800):
    new_y = generate_value(origin_train_y[i])
    new_y = new_y[::-1]
    encoder_value.append(new_y)

encoder_value = np.array(encoder_value)
np.save('encoder_value_y.npy', encoder_value)

new_train_x = np.concatenate((origin_train_x, encoder_value), axis=1)
np.save('new_train_x.npy', new_train_x)

new_train_y = np.ones(800)
np.save('new_train_y.npy', new_train_y)