# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Convert QubitOperator to string format."""

from mindquantum.core.operators import QubitOperator

def convert_ops(ops:QubitOperator):
    """Convert QubitOperator to easy convert format."""
    out = []
    for i, j in ops.terms.items():
        out.append([i, j.const])
    return out

if __name__ == '__main__':
    test_ops = QubitOperator('Z0 Y1') + QubitOperator('X3 Z1', 1.3) + QubitOperator("")
    print(convert_ops(test_ops))
