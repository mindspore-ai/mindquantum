# Copyright 2021 Huawei Technologies Co., Ltd
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

"""Circuit library."""

from .amplitude_encoder import amplitude_encoder
from .bitphaseflip_operator import bitphaseflip_operator
from .general_ghz_state import general_ghz_state
from .general_w_state import general_w_state
from .quantum_fourier import qft
from .qjpeg import qjpeg
from .qudit_mapping import qudit_symmetric_decoding, qudit_symmetric_encoding, qutrit_symmetric_ansatz, mat_to_op

__all__ = [
    'qft', 'qjpeg', 'amplitude_encoder', 'general_w_state', 'general_ghz_state', 'bitphaseflip_operator',
    'qudit_symmetric_decoding', 'qudit_symmetric_encoding', 'qutrit_symmetric_ansatz', 'mat_to_op'
]

__all__.sort()
