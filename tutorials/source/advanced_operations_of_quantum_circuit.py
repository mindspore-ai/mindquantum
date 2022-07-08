#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Example of advanced use of quantum circuits."""

from mindquantum import RX, RZ, SWAP, UN, Circuit, H, X
from mindquantum.algorithm.library import qft
from mindquantum.core.circuit import (
    add_prefix,
    apply,
    change_param_name,
    controlled,
    dagger,
    shift,
)

# controlled()
u1 = qft(range(2))  # 构建量子线路
print(u1)
u2 = controlled(u1)(2)  # 对线路添加控制量子位q2，返回一个新的线路
print(u2)

u3 = controlled(u1)

u4 = u3(2)
print(u4)

u5 = controlled(qft)
u5 = u5([2, 3], [0, 1])  # 批量添加控制位
print(u5)

# dagger()
u1 = qft(range(3))
print(u1)
u2 = dagger(u1)
print(u2)

u3 = dagger(qft)
u4 = u3(range(3))
print(u4)

# apply()
u1 = qft([0, 1])
circuit1 = apply(u1, [1, 0])  # 将量子线路u1作用在比特q1, q0上
print(circuit1, "\n")

u2 = apply(qft, [1, 0])  # 将qft作用在比特q0, q1上
circuit2 = u2([0, 1])
print(circuit2)

# add_prefix()
circ = Circuit().rx("theta", 0)
print(circ)

circ = add_prefix(circ, 'l0')  # 添加后，参数"theta"就变成了"l0_theta"
print(circ)


def u(qubit):
    """Qubit generator function."""
    return Circuit([H.on(0), RX('a').on(qubit)])


u1 = u(0)
u1 = add_prefix(u1, 'ansatz')
print(u1)

u2 = add_prefix(u, 'ansatz')
u2 = u2(0)
print(u2)


# change_param_name()

u1 = u(0)
u1 = change_param_name(u1, {'a': 'b'})
print(u1)

u2 = change_param_name(u, {'a': 'b'})
u2 = u2(0)
print(u2)

# UN()
circuit1 = Circuit()
circuit1 += UN(H, 4)  # 将H门作用在每一位量子比特上
print(circuit1)

circuit2 = UN(X, maps_obj=[0, 1], maps_ctrl=[2, 3])
print(circuit2)

circuit3 = UN(SWAP, maps_obj=[[0, 1], [2, 3]]).x(2, 1)
print(circuit3)

# shift()
circ = Circuit().x(1, 0)
print(circ)

circ = shift(circ, 1)
print(circ)  # 线路作用的量子比特从q0,q1变为q1,q2

# 搭建Encoder
template = Circuit([X.on(1, 0), RZ('alpha').on(1), X.on(1, 0)])
encoder = (
    UN(H, 4)
    + (RZ(f'{i}_alpha').on(i) for i in range(4))
    + sum(add_prefix(shift(template, i), f'{i + 4}') for i in range(3))
)
print(encoder)
encoder.summary()
