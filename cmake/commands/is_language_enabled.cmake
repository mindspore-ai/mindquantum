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

# Check if a language has been enabled without attempting to enable it
#
# is_language_enabled(<lang> <resultvar>)
#
# If the language <lang> has already been enabled, <resultvar> is set to TRUE. Otherwise it is set to FALSE.
function(is_language_enabled _lang _var)
  get_property(_supported_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if(NOT _lang IN_LIST _supported_languages)
    set(${_var}
        FALSE
        PARENT_SCOPE)
  else()
    set(${_var}
        TRUE
        PARENT_SCOPE)
  endif()
endfunction()
