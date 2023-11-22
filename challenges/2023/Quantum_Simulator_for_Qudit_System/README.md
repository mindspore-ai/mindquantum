# QudiTop

This project named QudiTop is a numerically quantum simulator for qudit system based on MindSpore. QudiTop includes a variety of qudit gates, circuit and expectation etc. Especially for qudit gate, we can easily define the objective qudits, control qudits and corresponding control states, which allow you to simulate complex control in qudit circuit. Besides, QudiTop supports the difference of parameters, which allows you to run VQE (variational quantum eigen-solver) application well.

## File Structure

The core source code lies in `quditop/` folder and examples in `examples/`.

```log
.
├── README.md
├── Math_for_Gates_and_Circuit.md      # The matrix of gates and so on.
├── setup.py
├── ...
├── examples
│   ├── demo_basic.ipynb    # Demonstrate basic functions.
│   └── demo_vqe.ipynb      # Demonstrate the VQE application.
└── quditop
    ├── circuit.py          # Define the Circuit class.
    ├── common.py           # Common function.
    ├── evolution.py        # Evolution of quantum state.
    ├── gates.py            # Quantum gates.
    ├── global_var.py       # Define some global configuration.
    └── utils.py            # Some tool functions.

```

## Install

To install this package:

- Clone the package: `git clone https://github.com/forcekeng/QudiTop.git`
- Run `python setup.py install`

Now this package is installed on your computer. To uninstall it, run `pip uninstall quditop` at terminal.

## Qudit Gate

These qudits gates are realized, the matrix definitions follows github repository [GhostArtyom/QuditVQE](https://github.com/GhostArtyom/QuditVQE/tree/main/QuditSim).

- 1-qudit gates
  - Extended Pauli gates: $X_d^{(i,j)},Y_d^{(i,j)},Z_d^{(i,j)}$
  - Rotation gate: $RX_d^{(i,j)},RY_d^{(i,j)},RZ_d^{(i,j)}$
  - Hadamard gate: $H_d$
  - Increment gate: $\mathrm{INC}_d$
  - Phase gate: $PH$

- Multi-qudits gates
  - Universal math gate: $UMG$, it can act on one or multi qudits, according to the shape of matrix.
  - SWAP gate: $SWAP$.
  - Controlled gate: Control any above 1-qudit or multi-qubits gates by one or multi controlled qudits.

For gates that contain parameters, such as $RX$, calculating the difference is supported to run VQE applications.
A detailed introduction and mathematical matrix can be seen in document [Math_for_Gates_and_Circuit](./Math_for_Gates_and_Circuit.md). Comparing with qubit, the qudit system is more complex, not only more state level has to be solved, but also the control states may change. In qubit system we normally use to the state $1$ as the control state, while in qudit system, such for $d=3$, there're 3 states ($0,1$ and $2$), we can use $1$ or $2$ as control state, or even
use state $0$. So a common format of gate is:

- `H(dim).on(obj_qudits, ctrl_qudits, ctrl_states)`
- `X(dim, ind).on(obj_qudits, ctrl_qudits, ctrl_states)`
- `RX(dim, ind, pr).on(obj_qudits, ctrl_qudits, ctrl_states)`

The `dim` is the dimension of the qudit, for qubit the dimension is 2, dimension is $3$ means there are 3 states for qudit.
The `Quditop` framework allows the `dim` is at most $10$, normally we use `dim`$=3,4$ or $5$ in qudit simulation.

## Qudit Circuit

 `class Circuit` is implemented to simulate the quantum circuit. Interfaces of `Circuit` include but not limit to:

- `append`, `+` and '+=' for `Circuit` and qudit gate(s).
- `as_encoder` and `as_ansatz` to allow or inhibit the difference of parameters.
- `get_qs` and `set_qs` to get or set the last state or set the initial state of circuit.

We can build a qudit circuit simply by

```python
dim = 2
n_qudits = 4

circ = Circuit(dim, n_qudits, gates=[
    H(dim).on(0),
    X(dim, [0,1]).on(0, [2,3], [1, 1])
])
```

## Hamiltonian

We don't need a single class for hamiltonian. Because essentially we can treat hamiltonian as a series of gates that act on last quantum state. Specifically, we can write a hamiltonian as

```python
# Assuming we act X and Y gates on qudit 1 and 2 respectively
xg = X(dim=3).on(1)
yg = Y(dim=3).on(2)
# a and b are the const coefficients of the hamiltonian
a = 1.0
b = 2.0
hams = [(a, xg), (b, yg)]
```

## Expectation

`class Expectation` is used to get the measure expectation with given hamiltonian and circuit. Mathematically, expectation $h = \langle\psi |H| \psi\rangle$, where $O$ is the hamiltonian and $\psi$ is the last quantum state.

```python
expect_fn = Expectation(hams)    # Define expectation
last_qs = circ()                 # Get the last state of circuit
out = expect_fn(last_qs)         # Get the expectation
```

## Difference of Parameter

This quantum simulator is realized based on PyTorch. Comparing with deep learning, you can just treat one qudit gate as a layer of network. In fact, the gate is subclass of `torch.nn.Module`. So you can easily use the `loss.backward()` to get the gradient of parameter, and `optimizer.step()` to update parameter with specific optimizer. If you never or new to PyTorch, the documents of PyTorch are recommended. You can see how to train a hybrid quantum-classical VQE (variational quantum eigensolver) in file `example/demo_vqe.ipynb`.

## Get Started

```python
from quditop.gates import H, X, Y, Z, RX, RY, RZ, UMG
from quditop.circuit import Circuit

dim = 2
n_qudits = 4

circ = Circuit(dim, n_qudits, gates=[
    H(dim).on(0),
    X(dim, [0,1]).on(0, [2,3], [1, 1]),
    Y(dim).on(1),
    Z(dim).on(1, 2, 1),
    RX(dim, pr='param').on(3),
    RY(dim, pr=1.0).on(2, 3, 1),
    X(dim).on(1, [0, 2, 3], 1),
    RY(dim, pr=2.0).on(3, [0, 1, 2], 1)
])

print(circ)
print(f"quantum state:\n{circ.get_qs(ket=True)}")
```

The output should like this:

```text
Circuit(
  (gates): ModuleList(
    (0): H(2|0)
    (1): X(2 [0 1]|0 <-: 2 3 - 1 1)
    (2): Y(2 [0 1]|1)
    (3): Z(2 [0 1]|1 <-: 2 - 1)
    (4): RX(2 [0 1] param|3)
    (5): RY(2 [0 1] _param_|2 <-: 3)
    (6): X(2 [0 1]|1 <-: 0 2 3 - 1 1 1)
    (7): RY(2 [0 1] _param_|3 <-: 0 1 2)
  )
)
quantum state:
0.6205j¦0010⟩
0.6205j¦0011⟩
0.2975¦1010⟩
0.2975¦1011⟩
0.1625¦1101⟩
0.1625¦1110⟩
```

More details in the folder of `examples/`.

## Thanks to

[1] [pytorch/pytorch: Tensors and Dynamic neural networks in Python with strong GPU acceleration (github.com)](https://github.com/pytorch/pytorch)

[2] [pyquil.simulation._numpy — pyQuil 4.0.3 documentation (rigetti.com)](https://pyquil-docs.rigetti.com/en/stable/_modules/pyquil/simulation/_numpy.html)

[3] [【开发者群英会】在MindQuantum中实现任意维度的通用qudit量子线路模拟 · Issue #I7Q1UV · MindSpore/community - Gitee.com](https://gitee.com/mindspore/community/issues/I7Q1UV)

[4] [QuditVQE/QuditSim at main · GhostArtyom/QuditVQE (github.com)](https://github.com/GhostArtyom/QuditVQE/tree/main/QuditSim)
