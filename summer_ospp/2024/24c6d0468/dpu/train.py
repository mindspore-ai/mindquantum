device = "npu"

# Download data from open datasets
from download import download
import mindspore
# mindspore.set_context(device_target="Ascend")
mindspore.set_context(device_target="CPU")
from mindspore.dataset import MnistDataset

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)

train_dataset = MnistDataset("MNIST_Data/train", shuffle=True).batch(50)
test_dataset = MnistDataset("MNIST_Data/test", shuffle=True).batch(50)


from tqdm import tqdm
from lsing import lsing_model

input_node_num=28*28
label_class_num=10
label_node_num=50
all_node_num=4264

from line_profiler import profile
@profile
def main():
    model = lsing_model()

    bar = tqdm(train_dataset.create_tuple_iterator())
    for epoch in range(100):
        acc = []
        bar = tqdm(train_dataset.create_tuple_iterator())
        for images_batch, labels_batch in bar:
            images_batch = images_batch.squeeze().reshape(-1, 28*28)
            labels_batch = labels_batch.squeeze().flatten().to(mindspore.int32)
            m = model.create_m(images_batch, labels_batch)
            m_data = model.construct(m, model.group_hidden)

            m = model.create_m(images_batch, labels_batch)
            m_model = model.construct(m, model.group_all)

            model.updateParams(m_data, m_model, batch_size=images_batch.shape[0])

            logits = m[:, input_node_num:input_node_num+label_node_num].reshape(-1, label_node_num//label_class_num, label_class_num)
            logits = logits.sum(axis=-2).argmax(axis=-1).to(mindspore.int32)
            logits = mindspore.numpy.where(logits==labels_batch, 1., 0.)
            acc = logits

            bar.set_postfix({
                "acc" : acc.mean()
            })
            
            exit()

main()