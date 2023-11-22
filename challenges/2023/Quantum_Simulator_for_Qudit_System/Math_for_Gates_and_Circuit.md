# Math for Gates and Circuit in QudiTop

QudiTop is a numerically quantum simulator for qudit system based on PyTorch. Here we provide some mathematical define of gates and the time evolution of quantum state.

## Qudit Gate

The define of gates comes from [GhostArtyom/QuditVQE](https://github.com/GhostArtyom/QuditVQE/blob/main/QuditSim/QuditSim.md).

- 1-qudit gates
  - Extended Pauli gates: $X_d^{(j,k)},Y_d^{(j,k)},Z_d^{(j,k)}$
  - Rotation gate: $RX_d^{(j,k)},RY_d^{(j,k)},RZ_d^{(j,k)}$
  - Hadamard gate: $H_d$
  - Increment gate: $\mathrm{INC}_d$
  - Phase gate: $PH$

- Multi-qudits gates
  - Universal math gate: $UMG$, it can act on one or multi qudits, according to the shape of matrix.
  - SWAP gate: $SWAP$.
  - Controlled gate: Control any above 1-qudit or multi-qubits gates by one or multi controlled qudits.

- $X, Y, Z, RX, RY, RZ$ Gate

```math
\begin{aligned}
&\sigma_x^{(j, k)}= |j\rangle\langle k| + |k\rangle\langle j|,\qquad

& X_d^{(j,k)} = \sigma_x^{(j, k)} + \sum_{l\ne j,k} |l\rangle \langle l|,\qquad

& RX_d^{(j,k)} = \exp\lbrace -{\mathrm i}\theta \sigma_x^{(j,k)}/2\rbrace\\

&\sigma_y^{(j, k)}= |j\rangle\langle k| + |k\rangle\langle j|,

& Y_d^{(j,k)} = \sigma_y^{(j, k)} + \sum_{l\ne j,k} |l\rangle \langle l|,\qquad

& RY_d^{(j,k)} = \exp\lbrace -{\mathrm i}\theta \sigma_y^{(j,k)}/2\rbrace\\

&\sigma_z^{(j, k)}= |j\rangle\langle k| + |k\rangle\langle j|,\qquad

& Z_d^{(j,k)} = \sigma_z^{(j, k)}  + \sum_{l\ne j,k} |l\rangle \langle l|,\qquad

& RZ_d^{(j,k)} = \exp\lbrace -{\mathrm i}\theta \sigma_x^{(j,k)}/2\rbrace\\
\end{aligned}
```

e.g., for $d=3$, we have

```math
X_3^{(0,1)}=\begin{pmatrix}
0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1
\end{pmatrix},\quad
Y_3^{(0,2)}=\begin{pmatrix}
0 & 0 & -\mathrm{i} \\ 0 & 1 & 0 \\ \mathrm{i} & 0 & 0
\end{pmatrix},\quad
Z_3^{(1,2)}=\begin{pmatrix}
1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1
\end{pmatrix}
```

```math
RX_3^{(0,1)}(\theta)=\begin{pmatrix}
\cos\frac{\theta}{2} & -\mathrm{i}\sin\frac{\theta}{2} & 0 \\
-\mathrm{i}\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
0 & 0 & 1
\end{pmatrix},\;\;
RY_3^{(0,2)}(\theta)=\begin{pmatrix}
\cos\frac{\theta}{2} & 0 & -\sin\frac{\theta}{2} \\ 0 & 1 & 0 \\ \sin\frac{\theta}{2} & 0 & \cos\frac{\theta}{2}
\end{pmatrix},\;\;
RZ_3^{(1,2)}(\theta)=\begin{pmatrix}
1 & 0 & 0 \\ 0 & \mathrm{e}^{-i\theta/2} & 0 \\ 0 & 0 & \mathrm{e}^{i\theta/2}
\end{pmatrix} \\
```

- Hadamard Gate

  - `H(dim).on(obj_qudits, ctrl_qudits, ctrl_states)`

```math
H_d\ket{j}=\frac{1}{\sqrt{d}}\sum_{i=0}^{d-1}\omega^{ij}\ket{i},\;\; \omega=\mathrm{e}^{2\pi\mathrm{i}/d},\;\;
(H_d)_{i,j}=\omega^{ij},\;\;
H_3=\frac{1}{\sqrt{3}}
\begin{pmatrix}
1 & 1 & 1 \\
1 & \omega^1 & \omega^{2} \\
1 & \omega^2 & \omega^{4}
\end{pmatrix}
```

- Increment Gate [1-3,9]

  - `INC(dim).on(obj_qudits, ctrl_qudits, ctrl_states)`

```math
\mathrm{INC}_d\ket{j}=\ket{(j+1)\bmod d}
=\begin{pmatrix}
& 1 \\ \mathbb{I}_{d-1}
\end{pmatrix},\quad
\mathrm{INC}_3=\begin{pmatrix}
0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0
\end{pmatrix}
```

- Swap Gate [7]
  - `SWAP(dim).on(obj_qudits=[i, j], ctrl_qudits, ctrl_states)`

```math
\mathrm{SWAP}_d\ket{i,j}=\ket{j,i},\quad
(\mathrm{SWAP}_d)_{i,j}=\left\{\begin{array}{c}
1, & j=(i\times d+\lfloor i/d\rfloor)\bmod d^2 \\
0, & \text{others}
\end{array}\right.
```

```math
\mathrm{SWAP}_3=\begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
```

- General control gate, when the control state is $|{m}\rangle$, apply $U$ to qudits, where $U$ is a single(e.g $X$) or multi-qudits (e.g $SWAP$) gate.
  - That is, supporting `.on(obj_qudits=i, ctrl_qudits=j, ctrl_states=m)` for all gates.

```math
C^m[U_d]\ket{i,j}=\left\{\begin{array}{c}
\ket{i}\otimes U_d\ket{j} & i=m \\
\ket{i,j} & i\ne m
\end{array}\right.
```

```math
C^m[U_d]=\ket{m}\bra{m}\otimes U_d+\sum_{i\ne m}\ket{i}\bra{i}\otimes\mathbb{I}_{d^2-d}
=\begin{pmatrix}
\mathbb{I}_{dm} & \\ & U_d \\ && \mathbb{I}_{d(d-m-1)}
\end{pmatrix}
```

For two qudits,

```math
C^{(m,n)}[U_{d^2}]=
 \ket{m}\bra{m}\otimes
 \ket{n}\bra{n}
 \otimes U_d+\sum_{i\ne m}\ket{i}\bra{i}\otimes\mathbb{I}_{d^2-d}
=\begin{pmatrix}
\mathbb{I}_{dm} & \\ & U_d \\ && \mathbb{I}_{d(d-m-1)}
\end{pmatrix}
```

## State Evolution

State evolution means how the state variate after acting on specific operations. $|\psi_t\rangle = U|\psi_0\rangle$.

In QudiTop, we represent the quantum state as a tensor whose dimension is $d$.
