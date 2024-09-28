import torch
from download import download
device="cuda"

from tqdm import tqdm
from lsing import lsing_model
from cnn import CNN_lsing
import numpy as np

input_node_num=3*3
label_class_num=2
label_node_num=10


def create_img_label(batch_size=50):
    img = torch.zeros((batch_size, 3, 3)).cuda()
    lab = torch.randint(0, 2, (batch_size, )).cuda()
    for i, l in enumerate(lab):
        if l == 0:
            img[i, 0, 0] = img[i, 1, 1] = img[i, 2, 2] = 1
        else:
            img[i, 0, 2] = img[i, 1, 1] = img[i, 2, 0] = 1
    return img.reshape(batch_size, -1), lab

from line_profiler import profile
@profile
def main():
    # model = lsing_model(label_class_num=10).to(device)
    # model = CNN_lsing(inputsize=28, kenrnal_size=5, stride=3, kenrnal_num=32, label_class_num=2, label_node_num=10).to(device)
    model = CNN_lsing(inputsize=3, kenrnal_size=2, stride=1, kenrnal_num=4, label_class_num=2, label_node_num=10).to(device)

    test_acc = []
    with torch.no_grad():
        max_acc = 0
        for epoch in range(200):
            # train
            images_batch, labels_batch = create_img_label()
            m = model.create_m(images_batch, labels_batch)
            m_data = model.construct(m, model.group_hidden)

            m = model.create_m(images_batch, labels_batch)
            m_model = model.construct(m, model.group_all)

            model.updateParams(m_data, m_model, images=images_batch, batch_size=images_batch.shape[0])

            # test
            images_batch, labels_batch = create_img_label(500)
            m = model.create_m(images_batch)
            m = model.construct(m, model.group_clssify)

            logits = m[:, -label_node_num:].reshape(-1, label_node_num//label_class_num, label_class_num)
            logits = logits.sum(dim=-2).argmax(dim=-1)
            logits = torch.where(logits==labels_batch, 1., 0.)
            acc = logits

            print(f"epoch {epoch} test result acc:{acc.mean().item()}")
            test_acc.append(acc.mean().item())
            if acc.mean().item() > max_acc:
                max_acc = acc.mean().item()
                torch.save(model, "model.pth")
            np.savetxt("test_acc.txt", test_acc)

main()