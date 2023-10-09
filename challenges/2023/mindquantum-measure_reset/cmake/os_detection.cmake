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

set(OS_NAME "Unknown")
set(OS_RELEASE 0)
if(MSYS)
  if("$ENV{MSYSTEM}" STREQUAL "MSYS")
    set(OS_NAME "MSYS-MSYS")
  elseif("$ENV{MSYSTEM}" STREQUAL "MINGW64")
    set(OS_NAME "MSYS-MINGW64")
  elseif("$ENV{MSYSTEM}" STREQUAL "CLANG64")
    set(OS_NAME "MSYS-CLANG64")
  elseif("$ENV{MSYSTEM}" STREQUAL "UCRT64")
    set(OS_NAME "MSYS-UCRT64")
  else()
    set(OS_NAME "MSYS-unsupported")
  endif()
elseif(MINGW)
  set(OS_NAME "MinGW")
elseif(CYGWIN)
  set(OS_NAME "Cygwin")
elseif(MSVC)
  set(OS_NAME "MSVC-like")
  set(COMPILER_IS_CLANG_CL FALSE)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(COMPILER_IS_CLANG_CL TRUE)
  endif()
elseif(UNIX)
  if(APPLE)
    set(OS_NAME "MacOS")
    find_program(sw_vers_exec sw_vers)
    if(sw_vers_exec)
      execute_process(
        COMMAND ${sw_vers_exec} -productVersion
        OUTPUT_VARIABLE _macos_version
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      set(OS_RELEASE "${_macos_version}")
      set(OS_RELEASE
          "${_macos_version}"
          CACHE STRING "OS version")
    endif()
  else()
    # Tested with:
    #
    # * ArchLinux (x86_64 & aarch64)
    # * CentOS
    # * Debian
    # * Fedora
    # * OpenSUSE (leap)

    set(_fname /etc/os-release)
    if(NOT EXISTS ${_fname})
      set(_fname /usr/lib/os-release)
      if(NOT EXISTS ${_fname})
        set(_fname)
      endif()
    endif()

    if(_fname)
      set(_regex "^ID=\"?([a-zA-Z]+)\"?")
      file(STRINGS ${_fname} OS_NAME_RAW REGEX ${_regex})
      string(REGEX MATCH ${_regex} OS_NAME_RAW ${OS_NAME_RAW})
      set(OS_NAME ${CMAKE_MATCH_1})

      set(OS_RELEASE 0)
      set(_regex "^VERSION_ID=\"?([0-9\\.]+)\"?")
      file(STRINGS ${_fname} OS_RELEASE_RAW REGEX ${_regex})
      if(OS_RELEASE_RAW)
        string(REGEX MATCH ${_regex} OS_RELEASE_RAW ${OS_RELEASE_RAW})
        set(OS_RELEASE ${CMAKE_MATCH_1})
      endif()
    elseif(NOT $ENV{NIX_STORE} STREQUAL "") # CMake 3.14: if(DEFINED ENV{NIX_STORE})
      set(OS_NAME "nix")
      set(OS_RELEASE 0)
      # Try to find out Nix version if possible
      find_program(_nix nix)
      if(_nix)
        execute_process(
          COMMAND ${_nix} --version
          OUTPUT_VARIABLE _nix_version
          OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        set(OS_RELEASE ${_nix_version})
      endif()
    endif()
  endif() # APPLE
endif() # UNIX

message(STATUS "Detected system: ${OS_NAME} (${OS_RELEASE})")

# ==============================================================================

message(STATUS "Detected processor: ${CMAKE_SYSTEM_PROCESSOR}")
if(MQ_SKIP_SYSTEM_PROCESSOR_DETECTION)
  # custom setup: required variables are passed through cache / CMake's command-line
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(X86_64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*")
  set(X86 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
  set(AARCH64 1)
else()
  message(WARNING "MindQuantum: unrecognized target processor configuration")
endif()

# ==============================================================================

include(ProcessorCount)
if(JOBS)
  set(_max_num_threads ${JOBS})
else()
  ProcessorCount(_max_num_threads)
endif()
message(STATUS "Max. number of parallel jobs for sub-commands: ${_max_num_threads}")

# ==============================================================================
