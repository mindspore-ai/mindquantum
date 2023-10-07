# ==============================================================================
#
# Copyright 2020 <Huawei Technologies Co., Ltd>
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

# ~~~
# Generate a custom ` binscope` target to call binscope on a list of targets.
#
#   gen_binscope_target(<target> [target ...])
# ~~~
function(gen_binscope_target)
  if(BINSCOPE_OUTPUT)
    set(_binscope_output "${BINSCOPE_OUTPUT}")
  else()
    set(_binscope_output "${PROJECT_BINARY_DIR}/binscope_output.xls")
  endif()

  set(_binscope_args)
  list(APPEND _binscope_args "-a")
  list(APPEND _binscope_args "-o")
  list(APPEND _binscope_args "${_binscope_output}")

  foreach(tgt ${ARGN})
    list(APPEND _binscope_args "-f")
    list(APPEND _binscope_args "$<TARGET_FILE:${tgt}>")
  endforeach()

  add_custom_target(
    binscope
    COMMAND ${binscope_exec} ${_binscope_args}
    COMMENT "Running binscope")
endfunction()
