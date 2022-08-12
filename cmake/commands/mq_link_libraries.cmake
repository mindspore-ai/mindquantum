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

include(is_language_enabled)

# ==============================================================================

# ~~~
# Add one or more link libraries to some of the language specific targets (see setup_languages())
#
# mq_link_libraries([NO_MQ_TARGET, NO_TRYCOMPILE_TARGET, NO_TRYCOMPILE_FLAGCHECK_TARGET]
#                    <library>, [<library>, ...])
#
# Always modify the <LANG>_mindquantum target. If any of TRYCOMPILE, TRYCOMPILE_FLAGCHECK is also specified, then modify
# the corresponding target.
# ~~~
function(mq_link_libraries)
  cmake_parse_arguments(PARSE_ARGV 0 MLL "TRYCOMPILE;TRYCOMPILE_FLAGCHECK" "" "")

  foreach(lang C CXX CUDA NVCXX DPCXX)
    is_language_enabled(${lang} _enabled)
    if(_enabled)
      target_link_libraries(${lang}_mindquantum INTERFACE "$<$<LINK_LANGUAGE:${lang}>:${MLL_UNPARSED_ARGUMENTS}>")
      if(MLL_TRYCOMPILE)
        target_link_libraries(${lang}_try_compile INTERFACE "$<$<LINK_LANGUAGE:${lang}>:${MLL_UNPARSED_ARGUMENTS}>")
      endif()
      if(MLL_TRYCOMPILE_FLAGCHECK)
        target_link_libraries(${lang}_try_compile_flagcheck
                              INTERFACE "$<$<LINK_LANGUAGE:${lang}>:${MLL_UNPARSED_ARGUMENTS}>")
      endif()
    endif()
  endforeach()
endfunction()
