# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# lint_cmake: -whitespace/indent

if(CMAKE_NVCXX_COMPILER_FORCED)
  # The compiler configuration was forced by the user. Assume the user has configured all compiler information.
  set(CMAKE_NVCXX_COMPILER_WORKS TRUE)
  return()
endif()

include(CMakeTestCompilerCommon)

# work around enforced code signing and / or missing executable target type
set(__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
if(_CMAKE_FEATURE_DETECTION_TARGET_TYPE)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE ${_CMAKE_FEATURE_DETECTION_TARGET_TYPE})
endif()

# Remove any cached result from an older CMake version. We now store this in CMakeCXXCompiler.cmake.
unset(CMAKE_NVCXX_COMPILER_WORKS CACHE)

# Try to identify the ABI and configure it into CMakeCXXCompiler.cmake
include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerABI.cmake)
# NB: use *.nvcpp in order to force underlying try_compile() to use NVHPC for compiling and linking
cmake_determine_compiler_abi(NVCXX ${CMAKE_CURRENT_LIST_DIR}/CMakeNVCXXCompilerABI.nvcpp)
if(CMAKE_NVCXX_ABI_COMPILED)
  # The compiler worked so skip dedicated test below.
  set(CMAKE_NVCXX_COMPILER_WORKS TRUE)
  message(STATUS "Check for working NVCXX compiler: ${CMAKE_NVCXX_COMPILER} - skipped")
endif()

# This file is used by EnableLanguage in cmGlobalGenerator to determine that the selected C++ compiler can actually
# compile and link the most basic of programs.   If not, a fatal error is set and cmake stops processing commands and
# will not generate any makefiles or projects.
if(NOT CMAKE_NVCXX_COMPILER_WORKS)
  PrintTestCompilerStatus("NVCXX")
  __testcompiler_settrycompiletargettype()
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testCXXCompiler.nvcpp "#ifndef __cplusplus\n"
       "# error \"The CMAKE_NVCXX_COMPILER is set to a C compiler\"\n" "#endif\n" "int main(){return 0;}\n")
  # Clear result from normal variable.
  unset(CMAKE_NVCXX_COMPILER_WORKS)
  # Puts test result in cache variable.
  message(
    STATUS
      "  try_compile(
    CMAKE_NVCXX_COMPILER_WORKS ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testCXXCompiler.nvcpp
    OUTPUT_VARIABLE __CMAKE_NVCXX_COMPILER_OUTPUT)
")
  try_compile(
    CMAKE_NVCXX_COMPILER_WORKS ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testCXXCompiler.nvcpp
    OUTPUT_VARIABLE __CMAKE_NVCXX_COMPILER_OUTPUT)
  # Move result from cache to normal variable.
  set(CMAKE_NVCXX_COMPILER_WORKS ${CMAKE_NVCXX_COMPILER_WORKS})
  unset(CMAKE_NVCXX_COMPILER_WORKS CACHE)
  __testcompiler_restoretrycompiletargettype()
  if(NOT CMAKE_NVCXX_COMPILER_WORKS)
    printtestcompilerresult(CHECK_FAIL "broken")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
         "Determining if the NVCXX compiler works failed with "
         "the following output:\n${__CMAKE_NVCXX_COMPILER_OUTPUT}\n\n")
    string(REPLACE "\n" "\n  " _output "${__CMAKE_NVCXX_COMPILER_OUTPUT}")
    message(
      FATAL_ERROR
        "The C++ compiler\n  \"${CMAKE_NVCXX_COMPILER}\"\n" "is not able to compile a simple test program.\nIt fails "
        "with the following output:\n  ${_output}\n\n" "CMake will not be able to correctly generate this project.")
  endif()
  printtestcompilerresult(CHECK_PASS "works")
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
       "Determining if the NVCXX compiler works passed with "
       "the following output:\n${__CMAKE_NVCXX_COMPILER_OUTPUT}\n\n")
endif()

# Try to identify the compiler features
include(${CMAKE_ROOT}/Modules/CMakeDetermineCompileFeatures.cmake)
cmake_determine_compile_features(NVCXX)

# Re-configure to save learned information.
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeNVCXXCompiler.cmake.in
               ${CMAKE_PLATFORM_INFO_DIR}/CMakeNVCXXCompiler.cmake @ONLY)
include(${CMAKE_PLATFORM_INFO_DIR}/CMakeNVCXXCompiler.cmake)

if(CMAKE_NVCXX_SIZEOF_DATA_PTR)
  foreach(_file ${CMAKE_NVCXX_ABI_FILES})
    include(${_file})
  endforeach()
  unset(CMAKE_NVCXX_ABI_FILES)
endif()

set(CMAKE_TRY_COMPILE_TARGET_TYPE ${__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE})
unset(__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE)
unset(__CMAKE_NVCXX_COMPILER_OUTPUT)
