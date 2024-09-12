import pickle
import numpy as np
from glob import glob
import torch
import torch.optim as optim


def inverse_compute_ber(solution, bits):
    ## convert the bits from sionna style to constellation style
    bits_constellation = 1 - np.concatenate([bits[..., 0::2], bits[..., 1::2]], axis=-1)
    num_bits_per_symbol = bits_constellation.shape[1]
        
    rb = num_bits_per_symbol//2
    bits_hat_inv = bits.copy()
    for i in range(0, num_bits_per_symbol - 1):
        bits_hat_inv[:, i + 1] = np.logical_xor(bits_hat_inv[:, i], bits_hat_inv[:, i + 1]).astype(int)
    index1 = np.nonzero(bits_hat_inv[:, rb - 1] == 1)[0]
    bits_hat_inv[index1, rb:] = 1 - bits_hat_inv[index1, rb:]
    bits_hat_inv[bits_hat_inv == 0] = -1
    bits_hat_inv = bits_hat_inv.T.copy()
    halfrow = bits_hat_inv.shape[0]//2
    bits_hat_inv = np.column_stack((bits_hat_inv[:halfrow, :], bits_hat_inv[halfrow:, :]))
    bits_hat_inv = bits_hat_inv.reshape(rb, 2, -1)
    return bits_hat_inv.reshape(bits_hat_inv.size, 1)
    

def approximate_s0(H, y, num_bits_per_symbol, snr):
    # the size of constellation
    M = 2**num_bits_per_symbol
    Nr, Nt = H.shape
    N = 2 * Nt
    rb = np.ceil(num_bits_per_symbol/2).astype('int')
    qam_var = 1/(2**(rb-2))*np.sum(np.linspace(1,2**rb-1, 2**(rb-1))**2)
    y_tilde = np.concatenate([y.real, y.imag])
    stack_s = []
    y_tilde_norm = (y_tilde/np.sqrt(1+10**(-snr/10))/qam_var + np.sqrt(M) - 1)/rb
    for i in range(rb):
        stack_s.append(y_tilde_norm/2**(rb - 1 - i))
    s = np.vstack(stack_s)
    s = s - np.ones([N*rb, 1])
    return s

dataset = []
filelist = glob('MLD_data/*.pickle')

for filename in filelist:
    data = pickle.load(open(filename, 'rb'))
    s = approximate_s0(data['H'], data['y'], data['num_bits_per_symbol'], data['SNR'])
    dataset.append([s, data['bits']])

w0 = torch.randn(1, requires_grad=True)
w1 = torch.randn(1, requires_grad=True)
w2 = torch.randn(1, requires_grad=True)
w3 = torch.randn(1, requires_grad=True)
optimizer = optim.SGD([w0, w1, w2, w3], lr=0.01)

# 训练过程
for epoch in range(1000):  # 迭代1000次
    for i in range(150):  # 遍历每一组数据
        s = dataset[i][0]
        x = dataset[i][1]
        x_m = inverse_compute_ber(s, x)
        s_torch = torch.tensor(s)
        x_torch = torch.tensor(x_m)
        # 前向传播计算预测值
        pred = torch.tanh(w3 + w2*torch.tanh(w0 + w1 * s_torch))

        # 计算损失函数
        loss = (x_torch - pred).T@(x_torch - pred)/s_torch.shape[0]

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 梯度清零
        optimizer.zero_grad()
        
    # 打印每个epoch的损失
    print(f'Epoch {epoch}, Loss: {loss.item()}')
        
# 输出最终的参数
print(f'Final w0: {w0.item()}, w1: {w1.item()}, w2: {w2.item()}, w3: {w3.item()}')