device = "npu"

# Download data from open datasets
from download import download
import mindspore
mindspore.set_context(device_target="Ascend")
# mindspore.set_context(device_target="CPU")
from mindspore import nn, ops, Tensor, CSRTensor, COOTensor
import numpy as np

from tqdm import tqdm
from cnn import CNN_lsing

def create_img_label(batch_size=50):
    img = ops.zeros((batch_size, 3, 3))
    lab = np.random.randint(0, 2, (batch_size, ))
    lab = Tensor(lab)
    for i, l in enumerate(lab):
        if l == 0:
            img[i, 0, 0] = img[i, 1, 1] = img[i, 2, 2] = 1
        else:
            img[i, 0, 2] = img[i, 1, 1] = img[i, 2, 0] = 1
    return img.reshape(batch_size, -1), lab

label_node_num=10
label_class_num=2

from line_profiler import profile
@profile
def main():
    model = CNN_lsing(inputsize=3, kenrnal_size=2, stride=1, kenrnal_num=4, label_class_num=2, label_node_num=10)
    test_acc = []
    for epoch in range(500):
        # train
        images_batch, labels_batch = create_img_label()

        m = model.create_m(images_batch, labels_batch)
        m_data = model.construct(m, model.group_hidden)

        m = model.create_m(images_batch, labels_batch)
        m_model = model.construct(m, model.group_all)

        model.updateParams(m_data, m_model, batch_size=images_batch.shape[0])

        # test
        images_batch, labels_batch = create_img_label(500)
        m = model.create_m(images_batch)
        m = model.construct(m, model.group_clssify)

        logits = m[:, -label_node_num:].reshape(-1, label_node_num//label_class_num, label_class_num)
        logits = logits.sum(axis=-2).argmax(axis=-1).to(mindspore.int32)
        logits = mindspore.numpy.where(logits==labels_batch, 1., 0.)
        acc = logits

        print(f"epoch {epoch} test result acc:{acc.mean().item()}")
        test_acc.append(acc.mean().item())
        if acc.mean().item() > max_acc:
            max_acc = acc.mean().item()
            mindspore.save(model, "model.pth")
        np.savetxt("test_acc.txt", test_acc)

main()