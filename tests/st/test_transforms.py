#   Copyright (c) 2020 Huawei Technologies Co.,ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Test the transforms in the hiqfermion module.
"""

from mindquantum.hiqfermion.transforms.transform import Transform
from mindquantum.ops import FermionOperator

def test_transform():
    """Test different transform.
    Note we need add the pyscfcalculator to build the Hamiltonian for the
    test of bravyi-kitaev superfast transform.
    """
    op1 = FermionOperator('1^')
    op_transform = Transform(op1)
    op1_jordan_wigner = op_transform.jordan_wigner()
    assert str(op1_jordan_wigner) == '0.5 [Z0 X1] +\n-0.5j [Z0 Y1] '

    op1_parity = op_transform.parity()
    assert str(op1_parity) == '0.5 [Z0 X1] +\n-0.5j [Y1] '

    op1_bravyi_kitaev = op_transform.bravyi_kitaev()
    assert str(op1_bravyi_kitaev) == '0.5 [Z0 X1] +\n-0.5j [Y1] '

    op1_ternary_tree = op_transform.ternary_tree()
    assert str(op1_ternary_tree) == '0.5 [X0 Z1] +\n-0.5j [Y0 X2] '
    #TODO bravyi_kitaev_superfast()
