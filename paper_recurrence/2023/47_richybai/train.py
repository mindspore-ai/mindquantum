import os
os.environ["OMP_NUM_THREADS"] = "1"

from src.nn import BinaryAccAgent, LossAgent, ForwardWithLoss, TrainOneStep
from src.data import load_mnist, extract_3and6, resize_image_batch, create_dataset, frqi_data_compression
from src.qnn import QNN
from src.nn import Network, count_params, HingeLoss
import mindspore as ms
import mindspore.context as context
from mindspore import nn
import numpy as np
import os
import time
from sklearn.model_selection import KFold

lr = 0.001
epochs = 10
ms.set_seed(2022)
context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def train(network, train_generator, test_generator):

    # loss = HingeLoss()
    loss = nn.MSELoss(reduction='mean')
    optimizer = nn.Adam(network.trainable_params(), learning_rate=lr)
        
    net_with_loss = ForwardWithLoss(network, loss)  
    train_one_step = TrainOneStep(net_with_loss, optimizer)

    loss_agent = LossAgent()
    acc_agent = BinaryAccAgent()
    
    train_epoch_loss = []
    train_epoch_acc = []
    test_epoch_loss = []
    test_epoch_acc = []

    # testing loss
    loss_agent.clear()
    for data in test_generator.create_dict_iterator():
        images = data['image']
        labels = data['label'].reshape(-1, 1)
        loss = net_with_loss(images, labels)  # 计算损失值
        loss_agent.update(loss)
    test_loss = loss_agent.eval()
    test_epoch_loss.append(test_loss)
    # testing acc
    acc_agent.clear()
    for data in test_generator.create_dict_iterator():
        images = data['image']
        labels = data['label'].reshape(-1, 1)
        logits = network(images)
        acc_agent.update(logits, labels)
    test_acc = acc_agent.eval()
    test_epoch_acc.append(test_acc)
    print(f"befor training: testing loss: {test_loss:8.6}, accuracy: {test_acc:5.4}")
    for epoch in range(epochs):
        t1 = time.time()
        # training update and loss
        loss_agent.clear()
        for k, data in enumerate(train_generator.create_dict_iterator()):
            # 这里索引出来的label的shape是(n,), 需要是(n, 1)才能和net的输出对的上,
            images = data['image']
            labels = data['label'].reshape(-1, 1) # 把label改成(n, 1)的shape, 和网络输出对准
            train_one_step(images, labels)   # 执行训练，并更新权重
            loss = net_with_loss(images, labels)  # 计算损失值
            loss_agent.update(loss)
        train_loss = loss_agent.eval()
        train_epoch_loss.append(train_loss)
        # training acc
        acc_agent.clear()
        for data in train_generator.create_dict_iterator():
            images = data['image']
            labels = data['label'].reshape(-1, 1)
            logits = network(images)
            acc_agent.update(logits, labels)
        train_acc = acc_agent.eval()
        train_epoch_acc.append(train_acc)

        # testing loss
        loss_agent.clear()
        for data in test_generator.create_dict_iterator():
            images = data['image']
            labels = data['label'].reshape(-1, 1)
            loss = net_with_loss(images, labels)  # 计算损失值
            loss_agent.update(loss)
        test_loss = loss_agent.eval()
        test_epoch_loss.append(test_loss)
        # testing acc
        acc_agent.clear()
        for data in test_generator.create_dict_iterator():
            images = data['image']
            labels = data['label'].reshape(-1, 1)
            logits = network(images)
            acc_agent.update(logits, labels)
        test_acc = acc_agent.eval()
        test_epoch_acc.append(test_acc)
        t2 = time.time()
        print(f"epoch: {epoch+1:2}, training loss: {train_loss:8.6}, accuracy: \
{train_acc:5.4}, testing loss: {test_loss:8.6}, accuracy: {test_acc:5.4}, time: {t2 - t1:4.2f}s")

    # np.save(data_path, {'train_loss': train_epoch_loss, 'train_acc': train_epoch_acc,
    #                     'test_loss': test_epoch_loss, 'test_acc': test_epoch_acc})
    # ms.save_checkpoint(QNN, model_path)
    return test_epoch_acc


if __name__ == "__main__":


    # 1.1 加载MNIST数据
    x_train_path='./mnist/train-images-idx3-ubyte.gz'
    y_train_path='./mnist/train-labels-idx1-ubyte.gz'
    x_test_path='./mnist/t10k-images-idx3-ubyte.gz'
    y_test_path='./mnist/t10k-labels-idx1-ubyte.gz'
    (x_train, y_train), (x_test, y_test)=load_mnist(x_train_path, y_train_path, 
                                                    x_test_path, y_test_path, 
                                                    normalize=True, one_hot=False)
    (x_train, y_train), (x_test, y_test) = extract_3and6(x_train, y_train, x_test, y_test)

    x_data = np.concatenate([x_train, x_test], axis=0)
    y_data = np.concatenate([y_train, y_test], axis=0)
    print("all data:\n", x_data.shape, y_data.shape)
    
    validation_dict = {'qnn1': [], 'qnn2': [], 'qnn3': [], 'cnn1': [], 'cnn2': []}

    KF = KFold(10, shuffle=True)
    for K, (train_index, test_index) in enumerate(KF.split(x_data)):
        print(f"--------------{K}-fold----------------")
        x_train, y_train = x_data[train_index], y_data[train_index]
        x_test, y_test = x_data[test_index], y_data[test_index]

        x_train8 = resize_image_batch(x_train, 8)
        x_train16 = resize_image_batch(x_train, 16)
        x_test8 = resize_image_batch(x_test, 8)
        x_test16 = resize_image_batch(x_test, 16)

        x_train8 = np.where(x_train8 > 0.5, 1., 0).astype(np.float32)
        x_train16 = np.where(x_train16 > 0.5, 1., 0).astype(np.float32)
        x_test8 = np.where(x_test8 > 0.5, 1., 0).astype(np.float32)
        x_test16 = np.where(x_test16 > 0.5, 1., 0).astype(np.float32)

        print("8x8 resolution:", x_train8.shape, x_test8.shape)
        print("16x16 resolution:", x_train16.shape, x_test16.shape)
        x_train8_compressed = frqi_data_compression(x_train8)
        x_train16_compressed = frqi_data_compression(x_train16)
        x_test8_compressed = frqi_data_compression(x_test8)
        x_test16_compressed = frqi_data_compression(x_test16)
    
        train8 = create_dataset(zip(x_train8, y_train))
        test8 = create_dataset(zip(x_test8, y_test))

        train8_compressed = create_dataset(zip(x_train8_compressed, y_train))
        test8_compressed = create_dataset(zip(x_test8_compressed, y_test))

        train16 = create_dataset(zip(x_train16, y_train))
        test16 = create_dataset(zip(x_test16, y_test))

        train16_compressed = create_dataset(zip(x_train16_compressed, y_train))
        test16_compressed = create_dataset(zip(x_test16_compressed, y_test))

        
        qnn1 = QNN(8, False, 12, "mqvector")
        print("qnn1: 8x8 no compression, 12 layers, parameters:", count_params(qnn1))
        result1 = train(qnn1, train8, test8)
        validation_dict['qnn1'].append(result1)

        qnn2 = QNN(8, True, 16, "mqvector")
        print("qnn2: 8x8 compression, 16 layers, parameters:", count_params(qnn2))
        result2 = train(qnn2, train8_compressed, test8_compressed)
        validation_dict['qnn2'].append(result2)
        
        cnn1 = Network(64)
        print("cnn1: 8x8 64-1-1, parameters:", count_params(cnn1))
        result3 = train(cnn1, train8, test8)
        validation_dict['cnn1'].append(result3)

        qnn3 = QNN(16, True, 42, "mqvector")
        print("qnn3: 16x16 compression, parameters:", count_params(qnn3))
        result4 = train(qnn3, train16_compressed, test16_compressed)
        validation_dict['qnn3'].append(result4)

        cnn2 = Network(256)
        print("cnn2: 16x16 256-1-1, parameters:", count_params(cnn2))
        result5 = train(cnn2, train16, test16)
        validation_dict['cnn2'].append(result5)

    np.save('validation_mse.npy', validation_dict)