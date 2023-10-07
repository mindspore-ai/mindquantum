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

# pylint: disable=duplicate-code

"""Parse argument."""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-sampling', help='number of samplings', type=int, default=100)
parser.add_argument('-b', '--batchs', help='batchs', type=int, default=1)
parser.add_argument(
    '-o',
    '--omp-num-threads',
    help='OMP_NUM_THREADS for mindquantum or set_intra_op_parallelism_threads for tensorflow',
    type=int,
    default=1,
)
parser.add_argument('-p', '--parallel-worker', help='parallel worker', type=int, default=1)
