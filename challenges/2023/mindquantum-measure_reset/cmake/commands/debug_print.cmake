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

# lint_cmake: -whitespace/indent

include_guard()

# ~~~
# Print a message only if some debug mode variable are set.
#
# debug_print(<mode>)
# ~~~
function(debug_print mode)
  if(ENABLE_CMAKE_DEBUG
     OR MINDQUANTUM_DEBUG
     OR mindquantum_DEBUG)
    math(EXPR _argc "${ARGC} - 1")
    foreach(_idx RANGE 1 ${_argc} 1)
      set(_msg "${ARGV${_idx}}")
      message(${mode} "${_msg}")
    endforeach()
  endif()
endfunction()
