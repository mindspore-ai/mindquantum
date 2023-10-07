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

include(nvhpc_helpers)

# cmake-lint: disable=E1120,R0912,R0915

# Set the CUDA architecture target properties for both CUDA and NVCXX languages
#
# target_cuda_architectures(<target> [PUBLIC|PRIVATE|INTERFACE] [LANG <lang>] <arch> [<arch> ...])
#
# Properly set the CUDA architecture target properties based on the linker language of <target>. In the case of NVCXX,
# this function will also set the appropriate ``COMPILE_OPTIONS`` target properties. This is mostly relevant for NVCXX <
# 21.3 where only one compute capability may be specified on the command line.
#
# <arch> are specified as numeric values (e.g. ``60`` for ``cc60``, etc.)
#
# The ``INTERFACE``, ``PUBLIC`` and ``PRIVATE`` keywords are required to specify the scope of the following arguments.
# ``PRIVATE`` and ``PUBLIC`` items will populate the ``COMPILE_OPTIONS`` property of ``<target>``.  ``PUBLIC`` and
# ``INTERFACE`` items will populate the ``INTERFACE_COMPILE_OPTIONS`` property of ``<target>``.
#
# NB: This function requires CMake >= 3.18
function(target_cuda_architectures target)
  cmake_parse_arguments(PARSE_ARGV 1 CUDA_ARCH "PUBLIC;PRIVATE;INTERFACE" "LANG" "")
  set(_cuda_archs ${CUDA_ARCH_UNPARSED_ARGUMENTS})
  set(_linker_lang ${CUDA_ARCH_LANG})

  set(_target_scope PUBLIC)
  if(CUDA_ARCH_PRIVATE)
    set(_target_scope PRIVATE)
  elseif(CUDA_ARCH_INTERFACE)
    set(_target_scope INTERFACE)
  endif()

  if(NOT _cuda_archs)
    message(FATAL_ERROR "Missing argument: CUDA_ARCHITECTURES!")
  endif()

  if(NOT CUDA_ARCH_LANG)
    get_target_property(_linker_lang ${target} LINKER_LANGUAGE)
  endif()

  if(NOT _linker_lang)
    message(FATAL_ERROR "Unable to determine the linker language for ${target} or no explicitly specified language")
  endif()

  if(NOT _linker_lang STREQUAL CUDA AND NOT _linker_lang STREQUAL NVCXX)
    message(FATAL_ERROR "Target ${target} with linker language ${_linker_lang} cannot have CUDA architectures "
                        "property set")
  endif()

  list(SORT _cuda_archs)

  if(_linker_lang STREQUAL NVCXX)
    # Avoid adding duplicate -gpu=ccXX for NVHPC < 21.3
    if(CMAKE_NVCXX_COMPILER_VERSION VERSION_LESS 21.3)
      list(SORT _cuda_archs COMPARE NATURAL) # NB: COMPARE NATURAL requires CMake >= 3.18
      list(GET _cuda_archs 0 _nvcxx_archs)
      message(WARNING "target_cuda_architecture(${target} ...)\n"
                      "NVHPC < 21.3, cannot have more than one -gpu=ccXX -> only using -gpu=cc${_nvcxx_archs}")
    else()
      set(_nvcxx_archs ${_cuda_archs})
    endif()

    get_target_property(_copts ${target} COMPILE_OPTIONS)
    if(_copts)
      separate_arguments(_copts)
    else()
      set(_copts)
    endif()

    get_target_property(_lopts ${target} LINK_OPTIONS)
    if(_lopts)
      separate_arguments(_lopts)
    else()
      set(_lopts)
    endif()

    foreach(_arch ${_nvcxx_archs})
      set(_copt "-gpu=cc${_arch}")
      set(_lopt "${_copt}")

      if(NVCXX_COMPILER_VERSION VERSION_LESS 21.3)
        nvhpc_extract_cc(_cc_cflags ${CMAKE_NVCXX_FLAGS})
        nvhpc_extract_cc(_cc_lflags ${CMAKE_NVCXX_LDFLAGS})

        if(_cc_cflags OR _copt IN_LIST _copts)
          message(
            WARNING "target_cuda_architecture(${target} ...)\n"
                    "NVHPC < 21.3, -gpu=ccXX present in CMAKE_NVCXX_FLAGS or ${_arch} already present for compilation")
          set(_copt)
        endif()
        if(_cc_lflags OR _lopt IN_LIST _lopts)
          message(
            WARNING "target_cuda_architecture(${target} ...)\n"
                    "NVHPC < 21.3, -gpu=ccXX present in CMAKE_NVCXX_LDFLAGS or ${_arch} already present for linking")
          set(_lopt)
        endif()
      endif()

      if(_copt)
        target_compile_options(${target} ${_target_scope} "$<$<COMPILE_LANGUAGE:NVCXX>:${_copt}>")
      endif()
      if(_lopt)
        target_link_options(${target} ${_target_scope} "$<$<COMPILE_LANGUAGE:NVCXX>:${_lopt}>")
      endif()
    endforeach()
  endif()

  # _linker_lang == CUDA || NVCXX
  set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES ${_cuda_archs})
endfunction()
