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

# ==============================================================================

# ~~~
# Store a list of strings in cache.
#
# If <name> is a list, its elements each represent a line of text.
#
# store_in_cache(<name>
#                <value>)
# ~~~
function(store_in_cache name value)
  string(REPLACE ";" ";@" value "${value}")
  string(REPLACE "\n" ";@" value "${value}")

  # cmake-lint: disable=C0103
  set(${name}
      "${value}"
      CACHE INTERNAL "")

  string(MD5 _hash ${name})
  set(${name}_hash
      "${_hash}"
      CACHE INTERNAL "Control hash for ${var}")
endfunction()

# ~~~
# Load a variable from cache that was saved with store_in_cache()
#
# load_from_cache(<name>
#                 <output_var>
#                 [PAD_LENGTH <pad_length>])
#
# NB: ';@' are replaced with '\n' followed by a number of spaces
# ~~~
function(load_from_cache name output_var)
  cmake_parse_arguments(PARSE_ARGV 2 LFC "" "PAD_LENGTH" "")
  if(LFC_PAD_LENGTH)
    string(REPEAT " " ${LFC_PAD_LENGTH} _padding)
  else()
    set(_padding)
  endif()

  if(NOT DEFINED CACHE{${name}})
    message(FATAL_ERROR "Cannot load inexistent cache variable: ${name}")
  endif()

  string(REPLACE ";@" "\n${_padding}" _tmp "${${name}}")

  set(${output_var}
      "${_tmp}"
      PARENT_SCOPE)
endfunction()
