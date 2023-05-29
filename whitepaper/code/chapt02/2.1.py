import numpy as np
import mindquantum as mq

# We support for different types
all_types = [mq.float32, mq.float64, mq.complex64, mq.complex128]

# convert from numpy and to numpy
mq_float64 = mq.to_mq_type(np.float64)
numpy_float64 = mq.to_np_type(mq.float64)
