import numpy as np
import mindspore as ms
from src.data_generate import Generator
from src.dense_ode_net import DenseODENet


def ode_solve(H, s0, t_start, t_end, steps):
    dim = H.shape[0]
    s = np.zeros((dim, 1), dtype=complex)
    s.real[:, 0] = s0[:dim]
    s.imag[:, 0] = s0[dim:]

    dt = (t_end - t_start) / steps
    print('dt: ', dt)

    ih = H * complex(0, -1)

    for idx in range(steps):
        k1 = np.dot(ih, s)
        k2 = np.dot(ih, s + k1 * dt / 2)
        k3 = np.dot(ih, s + k2 * dt / 2)
        k4 = np.dot(ih, s + dt * k3)
        s = s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return s


def compare_ode_raw():
    dim = 3
    t_list = np.arange(0.1, 20, 0.1)
    generator = Generator(dim=dim, t_list=t_list)
    h, data = generator.generate_trajectory(1)
    print(h)
    # (num, t_len, dim * 2)
    print(data)

    idx = 8
    state_0 = data[0, 0, :]
    state_1 = data[0, idx, :]

    for steps in range(1, 100, 1):
        ode_s = ode_solve(H=h, s0=state_0, t_start=0, t_end=t_list[idx - 1], steps=steps)
        raw_real = state_1[:dim]
        raw_imag = state_1[dim:]
        ode_real = ode_s.real[:, 0]
        ode_imag = ode_s.imag[:, 0]

        modulus = np.sum(np.square(raw_real) + np.square(raw_imag))
        difference = np.sum(np.square(ode_real - raw_real) + np.square(ode_imag - raw_imag))
        error_rate = np.sqrt(difference) / np.sqrt(modulus)
        print('steps: {} error rate: {}'.format(steps, error_rate))


h_dim = 2
t_list = np.arange(0.1, 20, 0.1)
ode_net = DenseODENet(depth=4, max_dt=0.01, h_dim=h_dim, init_range=[0.001, 0.01])
manual_w = ms.numpy.zeros((5, 5))

# manual_w[0, 1: 5] = 1
# manual_w[1, 1: 5] = ms.Tensor([1/2, 0, 0, 1/6])
# manual_w[2, 2: 5] = ms.Tensor([1/2, 0, 2/6])
# manual_w[3, 3: 5] = ms.Tensor([1, 2/6])
# manual_w[4, 4: 5] = ms.Tensor([1/6])

manual_w[0, 1: 5] = ms.Tensor([0.5243074,  0.517302,   0.00151536, 1.])
manual_w[1, 1: 5] = ms.Tensor([0.12954262, 0.12887144, 0.0010741,  0.44956145])
manual_w[2, 2: 5] = ms.Tensor([0.06948444, 0.00101975, 0.5297559])
manual_w[3, 3: 5] = ms.Tensor([0.00190731, 0.52713084])
manual_w[4, 4: 5] = ms.Tensor([0.])

ode_net.trainable_w.set_data(manual_w)

print(ode_net.dense_w())

generator = Generator(dim=h_dim)
H, data, t_points = generator.generate_trajectory(1, t_list=t_list)
s0 = ms.Tensor(data[:1, 0, :])
print('s0: ')
print(s0)

t_idx = 8
H = ms.Tensor(H)
net_s = ode_net.construct(H=H, S0=s0, T=ms.Tensor(t_points[t_idx]))
print(net_s)
print(data[0, t_idx, :])

