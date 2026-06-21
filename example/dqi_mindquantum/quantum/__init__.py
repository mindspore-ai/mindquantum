"""
量子电路模块。

MindQuantum quantum circuits for DQI.

提供基于 MindQuantum 的量子电路构建：
  - Dicke 态 (dicke_states): SCS 门、受控 Dicke 态制备
  - 权重编码 (weight_encoding): RY/CRY 级联、最优权重计算
  - 电路组件 (circuit_components): 相位 Z、综合征计算、Hadamard
  - 解码策略 (decoding): LUT 查找表、量子 BP1 解码器
  - 完整电路 (full_circuit): DQICircuit 类、run_dqi_histogram
"""

from .dicke_states import build_controlled_scs, build_controlled_dicke_prep
from .weight_encoding import (
    compute_optimal_w,
    build_embed_weights,
    build_uncompute_weight,
)
from .circuit_components import build_phase_z, build_syndrome_compute, build_hadamard
from .decoding import syndrome_LUT, build_lut_decode, build_quantum_bp1_circuit
from .full_circuit import DQICircuit, run_dqi_histogram

__all__ = [
    "build_controlled_scs",
    "build_controlled_dicke_prep",
    "compute_optimal_w",
    "build_embed_weights",
    "build_uncompute_weight",
    "build_phase_z",
    "build_syndrome_compute",
    "build_hadamard",
    "syndrome_LUT",
    "build_lut_decode",
    "build_quantum_bp1_circuit",
    "DQICircuit",
    "run_dqi_histogram",
]
