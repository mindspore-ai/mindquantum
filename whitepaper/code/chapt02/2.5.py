import numpy as np
import mindquantum as mq
from mindquantum.core.operators import QubitOperator, FermionOperator
from mindquantum.core.parameterresolver import ParameterResolver

# Construct a QubitOperator
q0 = QubitOperator('X0 Y1', ParameterResolver('a', dtype=mq.complex128))
# a [X0 Y1]
q0.dtype
# mindquantum.complex128

# Or simplify
q0 = QubitOperator('X0 Y1', 'a')
q0.dtype
# mindquantum.float64


# Construct a FermionOperator
f0 = FermionOperator('1^ 0', np.sqrt(2)) # '^' means dagger.
# √2 [1^ 0]

# Arithmetic operation
q1 = q0 * QubitOperator('Z1')
# (1j)*a [X0 X1]
q1.dtype
# mindquantum.complex128

q1 + 3.0
# (1j)*a [X0 X1] +
#      3 []

q1 - q0
# (1j)*a [X0 X1] +
#     -a [X0 Y1]

# Set the parameter value of operator
q2 = q1.subs({'a': 1.0})
# (1j) [X0 X1]

# Get matrix of operator
csr_matrix = q2.matrix()
csr_matrix.toarray()
# array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j],
#        [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j],
#        [0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j],
#        [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j]])
