import numpy as np
from mindspore import dataset as ds
from struct import unpack
import gzip
import os
from PIL import Image


def __read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    return img


def __read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab


def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img


def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def load_mnist(x_train_path, y_train_path, x_test_path, y_test_path, normalize=True, one_hot=True):
    
    '''读入MNIST数据集
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    Returns
    ----------
    (训练图像, 训练标签), (测试图像, 测试标签)
    '''
    image = {
        'train' : __read_image(x_train_path),
        'test'  : __read_image(x_test_path)
    }

    label = {
        'train' : __read_label(y_train_path),
        'test'  : __read_label(y_test_path)
    }
    
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def resize_image(image, resize=32, origin=28):
    """
    输入image, 要求是一维的向量, 符合MNIST数据最开始加载进来的形状28*28
    resize是改变之后的形状, 使用的插值方法是 BIlinear
    origin是原来的形状大小
    输出image也是一维的向量
    """
    img = Image.fromarray(image.reshape(origin, origin))
    imgBI = img.resize((resize, resize), Image.BILINEAR)
    imgBI = np.array(imgBI)
    return imgBI.reshape(-1)


def resize_image_batch(images, resize=32, origin=28):
    """
    输入的shape: [n, l]
    输出的shape: [n, resize * resize]
    """
    rst = []
    for image in images:
        rst.append(resize_image(image, resize, origin))
    return np.vstack(rst)


def extract_3and6(x_train, y_train, x_test, y_test):
    idx3 = np.array(np.where(y_train == 3)).squeeze()
    idx6 = np.array(np.where(y_train == 6)).squeeze()
    x_train36 = x_train[np.concatenate([idx3, idx6])]
    y_train36 = np.concatenate([np.ones_like(idx3), np.ones_like(idx6) * -1])

    idx3 = np.array(np.where(y_test == 3)).squeeze()
    idx6 = np.array(np.where(y_test == 6)).squeeze()
    x_test36 = x_test[np.concatenate([idx3, idx6])]
    y_test36 = np.concatenate([np.ones_like(idx3), np.ones_like(idx6) * -1])
    return (x_train36, y_train36), (x_test36, y_test36)
    

def frqi_data_compression(data):
    n_sample, n_pixel = data.shape
    data = data.copy().reshape(n_sample, int(n_pixel/4), 4)
    data[:, :, 1] += 1/4
    data[:, :, 2] += 1/2
    data[:, :, 3] += 3/4
    data_cos_sum = np.sum(np.cos(data*np.pi/2), axis=-1)
    data_sin_sum = np.sum(np.sin(data*np.pi/2), axis=-1)
    cos_new = np.arccos(np.sqrt(data_cos_sum**2 / (data_cos_sum**2 + data_sin_sum**2)))
    return cos_new


def create_dataset(data, batch_size=32, repeat_size=1):
    """
    输入zip在一起的x_data和y_label
    输出是 GeneratorDataset对象
    """
    input_data = ds.GeneratorDataset(list(data), column_names=['image', 'label'])
    input_data = input_data.batch(batch_size, drop_remainder=True)
    input_data = input_data.repeat(repeat_size)
    return input_data

