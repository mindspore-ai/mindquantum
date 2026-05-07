# ParameterResolver Reference

## Overview

`ParameterResolver` (PR) is MindQuantum's symbolic parameter system. It represents linear combinations of named parameters and is used throughout circuits, gates, and operators.

```python
from mindquantum.core.parameterresolver import ParameterResolver
```

## Creating ParameterResolvers

```python
# From string (single parameter with coefficient 1)
pr = ParameterResolver("alpha")  # alpha

# From dict (linear combination)
pr = ParameterResolver({"a": 2, "b": 0.5})  # 2*a + 0.5*b

# From number (constant)
pr = ParameterResolver(1.5)  # 1.5

# With constant term
pr = ParameterResolver({"a": 1}, const=0.3)  # a + 0.3
```

## Arithmetic Operations

```python
pr1 = ParameterResolver({"a": 1})
pr2 = ParameterResolver({"b": 2})

pr1 + pr2  # a + 2*b
pr1 * 3  # 3*a
pr1 + 0.5  # a + 0.5
pr1 - pr2  # a - 2*b
```

## Using PRs with Gates

When passing parameters to gates, MindQuantum accepts several formats:

```python
# String: creates PR with coefficient 1
RX("theta").on(0)

# Dict: creates PR with specified coefficients
RX({"theta": 2}).on(0)  # rotation = 2 * theta

# ParameterResolver directly
pr = ParameterResolver({"a": 0.5, "b": 1.0})
RX(pr).on(0)  # rotation = 0.5*a + b

# Number: fixed (non-parameterized) value
RX(1.57).on(0)
```

## Encoder / Ansatz Distinction

This is the fundamental pattern for hybrid quantum-classical models in MindQuantum. Every parameter in a circuit is either an **encoder** parameter (data input, not trainable) or an **ansatz** parameter (trainable weight).

### Marking at the Circuit Level

```python
# Mark entire circuit as encoder
encoder = Circuit().rx("x0", 0).ry("x1", 1)
encoder.as_encoder()

# Mark entire circuit as ansatz (default behavior)
ansatz = Circuit().ry("w0", 0).x(1, 0).ry("w1", 1)
ansatz.as_ansatz()

# Combine
full = encoder + ansatz
```

### Why This Matters

`Simulator.get_expectation_with_grad()` returns three values:

```python
f, g_enc, g_ans = grad_ops(encoder_data, ansatz_data)
# f:     expectation value           shape: [batch_size, n_hams]
# g_enc: gradient w.r.t encoder     shape: [batch_size, n_hams, n_enc_params]
# g_ans: gradient w.r.t ansatz      shape: [batch_size, n_hams, n_ans_params]
```

- **Encoder gradients** enable backprop through classical-to-quantum layers
- **Ansatz gradients** are used by optimizers (Adam, SGD) to update trainable weights
- **Encoder data** must be 2D: `np.array([[val1, val2, ...]])` — even for single samples

### Inspecting Parameters

```python
circ = encoder + ansatz
circ.encoder_params_name  # ['x0', 'x1']
circ.ansatz_params_name  # ['w0', 'w1']
circ.params_name  # ['x0', 'x1', 'w0', 'w1']
```

### No-Gradient Marking

For parameters that should not contribute gradients:

```python
encoder.no_grad()  # No gradient for encoder params during backward pass
```

## PRGenerator

For systematic parameter naming in large circuits:

```python
from mindquantum.core.parameterresolver import PRGenerator

prg = PRGenerator(prefix="layer0")
for i in range(4):
    circ += RY(prg.new()).on(i)
# Parameters: layer0_p0, layer0_p1, layer0_p2, layer0_p3
```

## Resolving Parameters

Apply numerical values to a parameterized object:

```python
# Resolve a circuit's parameters
circ = Circuit().rx("a", 0).ry("b", 1)
circ.get_qs(pr={"a": 0.5, "b": 1.0}, ket=True)

# Resolve a gate's matrix
gate = RX("theta")
gate.matrix({"theta": 3.14})

# In simulator operations
sim.apply_circuit(circ, pr={"a": 0.5, "b": 1.0})
sim.sampling(circ, pr={"a": 0.5, "b": 1.0}, shots=1000)
```
