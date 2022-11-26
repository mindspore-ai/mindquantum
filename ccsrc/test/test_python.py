
from . import _mq_densitymatrix as mqmatrix
from mindquantum import X

sim = mqmatrix(3, 42)
sim.apply_gate(X.on(0))
