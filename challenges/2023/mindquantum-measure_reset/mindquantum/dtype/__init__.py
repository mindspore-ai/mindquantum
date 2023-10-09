# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Data type module for MindQuantum."""

from .dtype import (
    complex64,
    complex128,
    float32,
    float64,
    is_double_precision,
    is_same_precision,
    is_single_precision,
    precision_str,
    to_complex_type,
    to_double_precision,
    to_mq_type,
    to_np_type,
    to_precision_like,
    to_real_type,
    to_single_precision,
)

__all__ = [
    "float32",
    "float64",
    "complex128",
    "complex64",
    "to_real_type",
    "to_complex_type",
    "to_double_precision",
    "to_single_precision",
    "to_mq_type",
    "to_np_type",
    "precision_str",
    "is_same_precision",
    "is_double_precision",
    "is_single_precision",
    'to_precision_like',
]
__all__.sort()
