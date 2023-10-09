# ==============================================================================
#
# Copyright 2022 <Huawei Technologies Co., Ltd>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

set(_find_utils_vars _pkg)

# ==============================================================================
# find_utils_push()

if(NOT DEFINED _find_utils_push_counter)
  set(_find_utils_push_counter 0)
endif()

math(EXPR _find_utils_push_counter "${_find_utils_push_counter}+1")

foreach(_var ${_find_utils_vars})
  set(_${_var}_${_find_utils_push_counter} ${${_var}})
endforeach()

# ==============================================================================
