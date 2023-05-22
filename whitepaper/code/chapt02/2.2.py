import numpy as np
import mindquantum as mq
from mindquantum.core.parameterresolver import ParameterResolver

# Basic usage
pr1 = ParameterResolver({'a': 1.0, 'b': 2.0 + 2.1j}, np.pi, dtype=mq.complex128) # dtype would be float64 by default
# ParameterResolver(dtype: complex128,
#   data: [
#          a: (1.000000, 0.000000),
#          b: (2.000000, 2.100000)
#   ],
#   const: (3.141593, 0.000000)
# )

print(pr1.expression())
# a + (2 + 2.1j)*b + π

# Another two constructors
pr2 = ParameterResolver('a') # A single variable: a
pr3 = ParameterResolver(1.2) # A constant number: 1.2

# Type conversion
pr4 = pr1.astype(mq.float64)
print(pr4)
# a + 2*b + π

# Set variable to encoder variable and no dot calculate gradient
pr4.encoder_part('a')
pr4.no_grad_part('a')
pr4.encoder_parameters
# ['a']
pr4.requires_grad_parameters
# ['b']
