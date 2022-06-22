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

include_guard()

include(debug_print)

# ~~~
# Set some variables within the MindQuantum's installed configuration file.
#
# mq_set(<var>
#         <value>)
#
# If a specified variable already exists, save its value and then restore it later by using mq_auto_unset.
# ~~~
macro(mq_set var value)
  list(APPEND _mq_variables ${var})
  if(DEFINED ${var})
    debug_print(STATUS "DEBUG: mq_set: ${var} defined, saving value to ${var}_old (${${var}})")
    set(${var}_old ${${var}})
  endif()
  if(NOT "${value}" STREQUAL "")
    debug_print(STATUS "DEBUG: mq_set: ${var} defined to ${value}")
    set(${var} "${value}")
  endif()
endmacro()

# ~~~
# Automatically unset any changes made by all previous calls to mq_set_variable()
#
# mq_unset_auto()
# ~~~
macro(mq_unset_auto)
  foreach(_var ${_mq_variables})
    if(DEFINED ${_var})
      debug_print(STATUS "DEBUG: mq_unset_auto: unsetting ${_var}")
      unset(${_var})
    endif()
    if(DEFINED ${_var}_old)
      debug_print(STATUS "DEBUG: mq_unset_auto: ${_var}_old defined, restoring value of ${_var} (${${_var}_old})")
      set(${_var} ${${_var}_old})
      unset(${_var}_old)
    endif()
  endforeach()
  unset(_var) # NB: not required anymore for CMake >= 3.21 (see CMP0124)
  unset(_mq_variables)
endmacro()
