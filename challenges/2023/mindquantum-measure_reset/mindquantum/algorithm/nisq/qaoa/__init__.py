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

"""Algorithm for quantum approximation optimization algorithm."""

from .max_2_sat_ansatz import Max2SATAnsatz
from .max_cut_ansatz import MaxCutAnsatz
from .max_cut_rqaoa_ansatz import MaxCutRQAOAAnsatz
from .qaoa_ansatz import QAOAAnsatz
from .rqaoa_ansatz import RQAOAAnsatz

__all__ = ['Max2SATAnsatz', 'MaxCutAnsatz', 'QAOAAnsatz', 'RQAOAAnsatz', 'MaxCutRQAOAAnsatz']
