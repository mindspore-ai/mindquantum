"""Some global variables."""

import mindspore

DTYPE = mindspore.float64
CDTYPE = mindspore.complex128
DEFAULT_VALUE = mindspore.Tensor(0.0, dtype=DTYPE)
DEFAULT_PARAM_NAME = '_param_'


__all__ = ['DTYPE', 'CDTYPE']
