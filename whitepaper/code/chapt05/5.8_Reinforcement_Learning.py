import gym
import time
import argparse
import numpy as np
from mindspore import context, Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from collections import deque
import random
from mindquantum.core import Circuit, UN, H, X, RZ, RX, RY, QubitOperator
from mindquantum.algorithm import HardwareEfficientAnsatz
from mindquantum.core import Hamiltonian
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator
from mindspore.nn import Adam
from mindspore import dtype as mstype
import matplotlib.pyplot as plt


class Memory(object):

    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(
                np.arange(len(self.buffer)), size=batch_size, replace=False
            )
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


class Actor(nn.Cell):

    def __init__(self):
        super(Actor, self).__init__()
        self.encoder = self.__get_encoder()
        self.ansatz = self.__get_ansatz()
        self.circuit = self.encoder + self.ansatz
        self.observable = self.__get_observable()
        self.sim = Simulator("mqvector", self.circuit.n_qubits)
        self.quantum_net = self.__get_quantum_net()
        self.softmax = ops.Softmax()

    def construct(self, x):
        x = self.quantum_net(x)
        x = self.softmax(x)
        return x

    def __get_encoder(self):
        encoder = Circuit()  # Initialize quantum circuit
        encoder += UN(H, 4)  # Apply H gate to each of the 4 qubits
        for i in range(4):  # i = 0, 1, 2, 3
            encoder += RY(f"alpha{i}").on(i)  # Apply RY(alpha_i) gate to qubit i
        encoder.as_encoder()
        encoder.summary()  # Summarize the Encoder
        return encoder

    def __get_ansatz(self):
        ansatz = HardwareEfficientAnsatz(
            4, single_rot_gate_seq=[RX, RY, RZ], entangle_gate=X, depth=3
        ).circuit  # Build Ansatz using HardwareEfficientAnsatz
        ansatz += X.on(2, 0)
        ansatz += X.on(3, 1)
        ansatz.as_ansatz()
        ansatz.summary()  # Summarize the Ansatz
        return ansatz

    def __get_observable(self):
        hams = [
            Hamiltonian(QubitOperator(f"Y{i}")) for i in [2, 3]
        ]  # Execute Pauli Y measurement on the 2nd and 3rd qubits
        print(hams)
        return hams

    def __get_quantum_net(self):
        ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        ms.set_seed(1)
        grad_ops = self.sim.get_expectation_with_grad(
            self.observable,
            self.circuit,
            None,
            None,
            parallel_worker=1,
        )
        QuantumNet = MQLayer(grad_ops)  # Build quantum neural network
        return QuantumNet

    def select_action(self, state):
        x = self.quantum_net(state)
        x = self.softmax(x)
        if np.random.rand() <= x[0][0]:
            action = 0
        else:
            action = 1
        return action


class Critic(nn.Cell):

    def __init__(self):
        super(Critic, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Dense(4, 64)
        self.fc2 = nn.Dense(64, 256)
        self.fc3 = nn.Dense(256, 1)

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MyActorLoss(nn.LossBase):
    """Define loss function"""

    def __init__(self, reduction="mean"):
        super(MyActorLoss, self).__init__(reduction)
        self.abs = ops.Abs()

    def construct(self, base, batch_action, old_p, advantage):
        prob = base[:, 0] * (1 - batch_action) + base[:, 1] * batch_action
        log_prob = ops.log(prob)
        old_prob = old_p[:, 0] * (1 - batch_action) + old_p[:, 1] * batch_action
        old_log_prob = ops.log(old_prob)
        ratio = ops.exp(log_prob - old_log_prob)
        L1 = ratio * advantage
        L2 = ratio.clip(0.9, 1.1) * advantage
        loss = 0 - ops.minimum(L1, L2)
        return self.get_loss(loss)


class MyWithLossActor(nn.Cell):
    """Define loss network"""

    def __init__(self, backbone, loss_fn):
        super(MyWithLossActor, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, batch_action, old_p, advantage):
        out = self.backbone(data)
        return self.loss_fn(out, batch_action, old_p, advantage)

    def backbone_network(self):
        return self.backbone


class MyWithLossCritic(nn.Cell):
    """Define loss network"""

    def __init__(self, backbone, loss_fn):
        super(MyWithLossCritic, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        out = self.backbone(data)
        return self.loss_fn(out, label)

    def backbone_network(self):
        return self.backbone


class MyTrainStep(nn.TrainOneStepCell):
    """Define training process"""

    def __init__(self, network, optimizer):
        """Parameter initialization"""
        super(MyTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, label):
        """Build training process"""
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)


class MyActorTrainStep(nn.TrainOneStepCell):
    """Define training process"""

    def __init__(self, network, optimizer):
        """Parameter initialization"""
        super(MyActorTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, batch_action, old_p, advantage):
        """Build training process"""
        weights = self.weights
        loss = self.network(data, batch_action, old_p, advantage)
        grads = self.grad(self.network, weights)(data, batch_action, old_p, advantage)
        return loss, self.optimizer(grads)


parser = argparse.ArgumentParser(description="MindSpore LeNet Example")
parser.add_argument(
    "--device_target", type=str, default="CPU", choices=["Ascend", "GPU", "CPU"]
)

args = parser.parse_known_args()[0]
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)


def train(N):
    starttime = time.time()
    env = gym.make("CartPole-v0")
    actor = Actor()
    old_actor = Actor()
    optim = Adam(actor.quantum_net.trainable_params(), learning_rate=1e-3)
    critic = Critic()
    value_optim = Adam(critic.trainable_params(), learning_rate=1e-4)
    gamma = 0.98
    lambd = 0.95
    # epsilon = 0.01
    memory = Memory(200)

    batch_size = 256

    loss_func_a = MyActorLoss()
    actor_with_criterion = MyWithLossActor(actor, loss_func_a)
    train_actor = MyActorTrainStep(actor_with_criterion, optim)

    loss_func_c = nn.loss.MSELoss()
    critic_with_criterion = MyWithLossCritic(critic, loss_func_c)
    # Construct training network
    train_critic = MyTrainStep(critic_with_criterion, value_optim)

    EPOCH = N
    re = []
    for epoch in range(EPOCH):
        state = env.reset()
        state = state[0]
        episode_reward = 0
        # Normalize each parameter to be between -π and π
        state = np.array(
            [
                state[0] * np.pi / 4.8,
                np.tanh(state[1]) * np.pi,
                state[2] * np.pi / (4.1887903e-01),
                np.tanh(state[3]) * np.pi,
            ]
        )
        for _ in range(200):
            state_tensor = Tensor([state])
            action = actor.select_action(state_tensor)
            next_state, reward, done, _, __ = env.step(action)
            next_state = np.array(
                [
                    next_state[0] * np.pi / 4.8,
                    np.tanh(next_state[1]) * np.pi,
                    next_state[2] * np.pi / (4.1887903e-01),
                    np.tanh(next_state[3]) * np.pi,
                ]
            )
            episode_reward += reward
            memory.add((state, next_state, action, reward, (done + 1) % 2))
            state = next_state
            if done:
                break

        old_actor.quantum_net.weight = actor.quantum_net.weight

        for _ in range(1):
            experiences = memory.sample(batch_size, True)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(
                *experiences
            )

            batch_state = Tensor([state.astype(np.float32) for state in batch_state])
            batch_next_state = Tensor(
                [next_state.astype(np.float32) for next_state in batch_next_state]
            )
            batch_action = Tensor([action for action in batch_action])
            batch_reward = Tensor([[reward] for reward in batch_reward])
            batch_done = Tensor([[done] for done in batch_done])

            old_p = old_actor(batch_state)
            value_target = batch_reward + batch_done * gamma * critic(batch_next_state)
            td = value_target - critic(batch_state)
            advantage = ops.zeros((len(td)), mstype.float32)
            for i in range(len(td)):
                temp = 0
                for j in range(len(td) - 1, i, -1):
                    temp = (temp + td[j][0]) * lambd * gamma
                temp += td[i][0]
                advantage[i] = temp

            train_actor(batch_state, batch_action, old_p, advantage)
            train_critic(batch_state, value_target)

        loss_val = critic_with_criterion(batch_state, value_target)
        print("epoch", epoch, " loss: ", loss_val)

        memory.clear()
        re.append(episode_reward)
        if (epoch + 1) % 1 == 0:
            print(
                "Epoch:{}/{}, episode reward is {}".format(
                    epoch + 1, EPOCH, episode_reward
                )
            )
            use = time.time() - starttime
            print("Have used ", use, " s, / ", use * EPOCH / (epoch + 1), "s")
    usetime = time.time() - starttime
    print("Time", usetime)

    plt.plot(re)
    plt.ylim(0, 200)
    plt.xlabel("Epoch")
    plt.ylabel("Score")

    plt.savefig("result_cartp2.jpg")
    return re


if __name__ == "__main__":
    train(1500)
