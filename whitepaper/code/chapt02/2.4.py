from mindquantum.core.circuit import Circuit
from mindquantum.core import gates as G
from mindquantum.core.parameterresolver import ParameterResolver

# Construct a quantum circuit
circ = Circuit()
circ += G.H.on(0)
circ += G.RX('a').on(1, 0)
# text form
# q0: ──H──────●────
#              │
# q1: ───────RX(a)──

# SVG picture if using jupyter notebook
circ.svg()

# Or save the svg picture to svg file
circ.svg().to_file('circ.svg')

# Another two constructors
# Construct from list
circ = Circuit([G.H.on(0), G.RX('a').on(1, 0)])

# Construct by chain low
circ = Circuit().h(0).rx('a', 1, 0)

# Set the parameterized quantum circuit to be a encoder circuit.
circ.encoder_params_name
# []

circ = circ.as_encoder()
circ.encoder_params_name
# ['a']

# Get the unitary matrix form of quantum circuit
circ.matrix(ParameterResolver({'a': 1.0}))
# array([[ 0.70710678+0.j        ,  0.70710678+0.j        ,
#          0.        +0.j        ,  0.        +0.j        ],
#        [ 0.62054458+0.j        , -0.62054458+0.j        ,
#          0.        -0.33900505j,  0.        +0.33900505j],
#        [ 0.        +0.j        ,  0.        +0.j        ,
#          0.70710678+0.j        ,  0.70710678+0.j        ],
#        [ 0.        -0.33900505j,  0.        +0.33900505j,
#          0.62054458+0.j        , -0.62054458+0.j        ]])

# Or simplify
circ.matrix({'a': 1.0})

# Get final state of quantum circuit
print(circ.get_qs(ket=True, pr={'a': 1.0})) # 'ket=True' means get quantum state in ket form.
# √2/2¦00⟩
# 0.6205445805637456¦01⟩
# -0.33900504942104487j¦11⟩

# Advance operator for circuit

from mindquantum.core.circuit import apply, add_prefix

# Apply circuit to other qubit
apply(circ, [1, 3])
# q1: ──H──────●────
#              │
# q3: ───────RX(a)──

# Add a prefix to all parameters
add_prefix(circ, 'l0')
# q0: ──H───────●──────
#               │
# q1: ───────RX(l0_a)──
