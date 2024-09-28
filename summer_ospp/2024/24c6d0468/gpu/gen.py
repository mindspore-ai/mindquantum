import torch
from download import download
device="cuda"

import matplotlib.pyplot as plt
import numpy as np

from read_MNIST import decode_idx3_ubyte as read_images
from read_MNIST import decode_idx1_ubyte as read_labels
images = torch.tensor(read_images("./MNIST_Data/train/train-images-idx3-ubyte")).reshape(60000, 28*28).to(device)
labels = torch.tensor(read_labels("./MNIST_Data/train/train-labels-idx1-ubyte")).to(device)
idxs = torch.where(torch.logical_xor(labels==0, labels==7))
images = images[idxs]
labels = labels[idxs]
labels = torch.where(labels==7, 1, 0)
train_dataset = torch.utils.data.TensorDataset(images, labels)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True, pin_memory=False)

images = torch.tensor(read_images("./MNIST_Data/test/t10k-images-idx3-ubyte")).reshape(10000, 28*28).to(device)
labels = torch.tensor(read_labels("./MNIST_Data/test/t10k-labels-idx1-ubyte")).to(device)
idxs = torch.where(torch.logical_xor(labels==0, labels==7))
images = images[idxs]
labels = labels[idxs]
labels = torch.where(labels==7, 1, 0)
test_dataset = torch.utils.data.TensorDataset(images, labels)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=True, pin_memory=False)


from tqdm import tqdm
from lsing import lsing_model

input_node_num=28*28
label_class_num=10
label_node_num=50
all_node_num=4264

from line_profiler import profile
@profile
def main():
    model = lsing_model(label_class_num=2).to(device)
    with torch.no_grad():
        num = 0
        for images_batch, labels_batch in test_dataset_loader:
            m = model.create_m(labels_batch)
            m = model.construct(m, model.group_gen)

            m = m[:, model.input_node_num]

            for i, image in enumerate(m):
                plt.imshow(np.array(image), cmap='Greys')
                plt.savefig(f'gen/num_{labels_batch[i]}.png')
main()