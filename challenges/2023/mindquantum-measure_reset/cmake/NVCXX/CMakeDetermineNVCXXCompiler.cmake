# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# lint_cmake: -whitespace/indent

# determine the compiler to use for C++ programs NOTE, a generator may set CMAKE_NVCXX_COMPILER before loading this file
# to force a compiler. use environment variable CXX first if defined by user, next use the cmake variable
# CMAKE_GENERATOR_NVCXX which can be defined by a generator as a default compiler If the internal cmake variable
# _CMAKE_TOOLCHAIN_PREFIX is set, this is used as prefix for the tools (e.g. arm-elf-g++, arm-elf-ar etc.)
#
# Sets the following variables: CMAKE_NVCXX_COMPILER CMAKE_COMPILER_IS_GNUCXX CMAKE_AR CMAKE_RANLIB
#
# If not already set before, it also sets _CMAKE_TOOLCHAIN_PREFIX

include(${CMAKE_ROOT}/Modules/CMakeDetermineCompiler.cmake)

if(${CMAKE_GENERATOR} MATCHES "Visual Studio")

elseif("${CMAKE_GENERATOR}" MATCHES "Green Hills MULTI")

else()
  if(NOT CMAKE_NVCXX_COMPILER)
    set(CMAKE_NVCXX_COMPILER_INIT NOTFOUND)

    # prefer the environment variable CXX
    if(NOT $ENV{NVCXX} STREQUAL "")
      get_filename_component(CMAKE_NVCXX_COMPILER_INIT $ENV{NVCXX} PROGRAM PROGRAM_ARGS CMAKE_NVCXX_FLAGS_ENV_INIT)
      if(CMAKE_NVCXX_FLAGS_ENV_INIT)
        set(CMAKE_NVCXX_COMPILER_ARG1
            "${CMAKE_NVCXX_FLAGS_ENV_INIT}"
            CACHE STRING "Arguments to CXX compiler")
      endif()
      if(NOT EXISTS ${CMAKE_NVCXX_COMPILER_INIT})
        message(
          FATAL_ERROR
            "Could not find compiler set in environment variable NVCXX:\n$ENV{NVCXX}.\n${CMAKE_NVCXX_COMPILER_INIT}")
      endif()
    endif()

    # next prefer the generator specified compiler
    if(CMAKE_GENERATOR_NVCXX)
      if(NOT CMAKE_NVCXX_COMPILER_INIT)
        set(CMAKE_NVCXX_COMPILER_INIT ${CMAKE_GENERATOR_NVCXX})
      endif()
    endif()

    # finally list compilers to try
    if(NOT CMAKE_NVCXX_COMPILER_INIT)
      set(CMAKE_NVCXX_COMPILER_LIST nvc++)
    endif()

    _cmake_find_compiler(NVCXX)
  else()
    _cmake_find_compiler_path(NVCXX)
  endif()
  mark_as_advanced(CMAKE_NVCXX_COMPILER)

  # Each entry in this list is a set of extra flags to try adding to the compile line to see if it helps produce a valid
  # identification file.
  set(CMAKE_NVCXX_COMPILER_ID_TEST_FLAGS_FIRST)
  set(CMAKE_NVCXX_COMPILER_ID_TEST_FLAGS
      # Try compiling to an object file only.
      "-c"
      # IAR does not detect language automatically
      "--c++"
      "--ec++"
      # ARMClang need target options
      "--target=arm-arm-none-eabi -mcpu=cortex-m3"
      # MSVC needs at least one include directory for __has_include to function, but custom toolchains may run MSVC with
      # no INCLUDE env var and no -I flags. Also avoid linking so this works with no LIB env var.
      "-c -I__does_not_exist__")
endif()

if(CMAKE_NVCXX_COMPILER_TARGET)
  set(CMAKE_NVCXX_COMPILER_ID_TEST_FLAGS_FIRST "-c --target=${CMAKE_NVCXX_COMPILER_TARGET}")
endif()

# Build a small source file to identify the compiler.
if(NOT CMAKE_NVCXX_COMPILER_ID_RUN)
  set(CMAKE_NVCXX_COMPILER_ID_RUN 1)

  # Try to identify the compiler.
  set(CMAKE_NVCXX_COMPILER_ID)
  set(CMAKE_NVCXX_PLATFORM_ID)
  file(READ ${CMAKE_ROOT}/Modules/CMakePlatformId.h.in CMAKE_NVCXX_COMPILER_ID_PLATFORM_CONTENT)

  set(CMAKE_NVCXX_COMPILER_ID_TOOL_MATCH_INDEX 2)

  include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerId.cmake)
  # NB: CMakeNVCXXCompilerId.cpp must be found in CMAKE_MODULE_PATH
  cmake_determine_compiler_id(NVCXX NVCXXFLAGS CMakeNVCXXCompilerId.cpp)

  # NB: -print-sysroot is anyway not supported by NVHPC

  # _cmake_find_compiler_sysroot(NVCXX)
endif()

# cmake-lint: disable=C0103

if(CMAKE_NVCXX_COMPILER_SYSROOT)
  string(CONCAT _SET_CMAKE_NVCXX_COMPILER_SYSROOT
                "set(CMAKE_NVCXX_COMPILER_SYSROOT \"${CMAKE_NVCXX_COMPILER_SYSROOT}\")\n"
                "set(CMAKE_COMPILER_SYSROOT \"${CMAKE_NVCXX_COMPILER_SYSROOT}\")")
else()
  set(_SET_CMAKE_NVCXX_COMPILER_SYSROOT "")
endif()

if(CMAKE_NVCXX_COMPILER_ARCHITECTURE_ID)
  set(_SET_CMAKE_NVCXX_COMPILER_ARCHITECTURE_ID
      "set(CMAKE_NVCXX_COMPILER_ARCHITECTURE_ID ${CMAKE_NVCXX_COMPILER_ARCHITECTURE_ID})")
else()
  set(_SET_CMAKE_NVCXX_COMPILER_ARCHITECTURE_ID "")
endif()

if(MSVC_NVCXX_ARCHITECTURE_ID)
  set(SET_MSVC_NVCXX_ARCHITECTURE_ID "set(MSVC_NVCXX_ARCHITECTURE_ID ${MSVC_NVCXX_ARCHITECTURE_ID})")
endif()

if(CMAKE_NVCXX_XCODE_ARCHS)
  set(SET_CMAKE_XCODE_ARCHS "set(CMAKE_XCODE_ARCHS \"${CMAKE_NVCXX_XCODE_ARCHS}\")")
endif()

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeNVCXXCompiler.cmake.in
               ${CMAKE_PLATFORM_INFO_DIR}/CMakeNVCXXCompiler.cmake @ONLY)

set(CMAKE_NVCXX_COMPILER_ENV_VAR "NVCXX")
