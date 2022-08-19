#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Utils for experimental."""

from ._mindquantum_cxx.ops import TermValue as TermValue_

TermValue = {
    'I': int(TermValue_.I),
    'X': int(TermValue_.X),
    'Y': int(TermValue_.Y),
    'Z': int(TermValue_.Z),
    'a': int(TermValue_.a),
    'adg': int(TermValue_.adg),
}


TermValueCpp = {
    'I': TermValue_.I,
    'X': TermValue_.X,
    'Y': TermValue_.Y,
    'Z': TermValue_.Z,
    'a': TermValue_.a,
    'adg': TermValue_.adg,
}


TermValueStr = {
    int(TermValue_.I): 'I',
    int(TermValue_.X): 'X',
    int(TermValue_.Y): 'Y',
    int(TermValue_.Z): 'Z',
    int(TermValue_.a): 'a',
    int(TermValue_.adg): 'adg',
}
