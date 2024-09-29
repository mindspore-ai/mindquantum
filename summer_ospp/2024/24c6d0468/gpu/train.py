import torch
from download import download
device="cuda"

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)

from read_MNIST import decode_idx3_ubyte as read_images
from read_MNIST import decode_idx1_ubyte as read_labels
images = torch.tensor(read_images("./MNIST_Data/train/train-images-idx3-ubyte")).reshape(60000, 28*28).to(device)
labels = torch.tensor(read_labels("./MNIST_Data/train/train-labels-idx1-ubyte")).to(device)
# idxs = torch.where(torch.logical_xor(labels==0, labels==7))
# images = images[idxs]
# labels = labels[idxs]
# labels = torch.where(labels==7, 1, 0)
train_dataset = torch.utils.data.TensorDataset(images, labels)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True, pin_memory=False)

images = torch.tensor(read_images("./MNIST_Data/test/t10k-images-idx3-ubyte")).reshape(10000, 28*28).to(device)
labels = torch.tensor(read_labels("./MNIST_Data/test/t10k-labels-idx1-ubyte")).to(device)
# idxs = torch.where(torch.logical_xor(labels==0, labels==7))
# images = images[idxs]
# labels = labels[idxs]
# labels = torch.where(labels==7, 1, 0)
test_dataset = torch.utils.data.TensorDataset(images, labels)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=True, pin_memory=False)


from tqdm import tqdm
from lsing import lsing_model
import numpy as np

input_node_num=28*28
label_class_num=10
label_node_num=50
all_node_num=4264

from line_profiler import profile
@profile
def main():
    model = lsing_model(label_class_num=10).to(device)

    test_acc = []
    with torch.no_grad():
        for epoch in range(50):
            acc = torch.tensor([]).cuda()
            bar = tqdm(train_dataset_loader)
            for images_batch, labels_batch in bar:
                m = model.create_m(images_batch, labels_batch)
                m_data = model.construct(m, model.group_hidden)

                m = model.create_m(images_batch, labels_batch)
                m_model = model.construct(m, model.group_all)

                model.updateParams(m_data, m_model, batch_size=images_batch.shape[0])

                logits = m[:, input_node_num:input_node_num+label_node_num].reshape(-1, label_node_num//label_class_num, label_class_num)
                logits = logits.sum(dim=-2).argmax(dim=-1)
                logits = torch.where(logits==labels_batch, 1., 0.)
                acc = torch.cat([acc, logits])

                bar.set_postfix({
                    "acc" : acc.mean().item()
                })

            for images_batch, labels_batch in test_dataset_loader:
                m = model.create_m(images_batch)
                m = model.construct(m, model.group_clssify)

                acc = torch.tensor([]).cuda()
                logits = m[:, input_node_num:input_node_num+label_node_num].reshape(-1, label_node_num//label_class_num, label_class_num)
                logits = logits.sum(dim=-2).argmax(dim=-1)
                logits = torch.where(logits==labels_batch, 1., 0.)
                acc = torch.cat([acc, logits])

            print(f"epoch {epoch} test result acc:{acc.mean().item()}")
            test_acc.append(acc.mean().item())
            torch.save(model, "model.pth")
            np.savetxt("test_acc.txt", test_acc)

main()