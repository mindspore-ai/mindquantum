# ==============================================================================
#
# Copyright 2022 <Huawei Technologies Co., Ltd>
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
#
# ==============================================================================

# lint_cmake: -whitespace/indent

# cmake-lint: disable=C0103,E1122

set(TWEEDLEDUM_PYBINDS OFF)
set(TWEEDLEDUM_USE_EXTERNAL_FMT OFF)

if(TWEEDLEDUM_DIR)
  message(STATUS "Using Tweedledum from external directory: ${TWEEDLEDUM_DIR}")
  list(APPEND CMAKE_MODULE_PATH ${TWEEDLEDUM_DIR}/cmake)
  add_subdirectory(${TWEEDLEDUM_DIR} ${CMAKE_CURRENT_BINARY_DIR}/tweedledum EXCLUDE_FROM_ALL)
else()
  if(ENABLE_GITEE)
    set(GIT_URL "https://gitee.com/donghufeng/tweedledum.git")
  else()
    set(GIT_URL "https://github.com/boschmitt/tweedledum.git")
  endif()
  set(GIT_TAG "e73beb23a3feeba02a851e3f8131e3c85a29de2b")

  set(CMAKE_OPTION
      -DBUILD_TESTING=OFF
      -DTWEEDLEDUM_EXAMPLES=OFF
      -DTWEEDLEDUM_USE_EXTERNAL_PYBIND11=ON
      -DTWEEDLEDUM_PYBINDS=OFF
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DCMAKE_CXX_EXTENSIONS=OFF
      -DPython_EXECUTABLE=${Python_EXECUTABLE}
      -DPython_VERSION=${Python_VERSION})

  if(MSVC)
    set(tweedledum_CXXFLAGS "/Zc:__cplusplus /EHsc /D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS /DWIN32_LEAN_AND_MEAN")
    if(ENABLE_ITERATOR_DEBUG)
      set(tweedledum_CFLAGS "/D_ITERATOR_DEBUG_LEVEL=${MQ_ITERATOR_DEBUG}")
      set(tweedledum_CXXFLAGS "${tweedledum_CXXFLAGS} /D_ITERATOR_DEBUG_LEVEL=${MQ_ITERATOR_DEBUG}")
    endif()
    list(APPEND CMAKE_OPTION -DCMAKE_DEBUG_POSTFIX=d)

    if(CMAKE_MT)
      list(APPEND CMAKE_OPTION -DCMAKE_MT=${CMAKE_MT})
    endif()
  elseif(MSYS AND NOT "${OS_NAME}" STREQUAL "MSYS-MSYS")
    set(tweedledum_CFLAGS "-DNT64")
    set(tweedledum_CXXFLAGS "-DNT64")
    mq_add_compile_definitions(NT64)

    find_package(Python 3.6.0 COMPONENTS Development.Embed) # NB: requires CMake >= 3.18
    list(APPEND CMAKE_OPTION -DPYTHON_LIBRARY=${Python_LIBRARIES})
  elseif("${OS_NAME}" STREQUAL "MinGW")
    set(tweedledum_CFLAGS "-DNT64")
    set(tweedledum_CXXFLAGS "-DNT64")
    mq_add_compile_definitions(NT64)
  endif()

  set(tweedledum_LOCAL_EXTRA_DEFINES)
  set(tweedledum_SYSTEM_EXTRA_DEFINES)

  if(POSIX_C_SOURCE)
    list(APPEND tweedledum_LOCAL_EXTRA_DEFINES TARGET tweedledum::tweedledum "_POSIX_C_SOURCE=${POSIX_C_SOURCE}")
    list(APPEND tweedledum_SYSTEM_EXTRA_DEFINES TARGET tweedledum::tweedledum "_POSIX_C_SOURCE=${POSIX_C_SOURCE}")
    set(tweedledum_CFLAGS "-D_POSIX_C_SOURCE=${POSIX_C_SOURCE}")
    set(tweedledum_CXXFLAGS "-D_POSIX_C_SOURCE=${POSIX_C_SOURCE}")
  endif()

  set(PATCHES
      ${CMAKE_CURRENT_LIST_DIR}/patch/cmake.patch001
      ${CMAKE_CURRENT_LIST_DIR}/patch/cxx20_compatibility.patch002
      ${CMAKE_CURRENT_LIST_DIR}/patch/parametric_ops_compatibility.patch003
      ${CMAKE_CURRENT_LIST_DIR}/patch/rxx_matrix_fix.patch004
      ${CMAKE_CURRENT_LIST_DIR}/patch/sync_bool_compare_and_swap.patch005
      ${CMAKE_CURRENT_LIST_DIR}/patch/misc_changes.patch006
      ${CMAKE_CURRENT_LIST_DIR}/patch/fix-msvc-platform-detection.patch007)

  if(NOT MQ_CXX_HAS_STD_LAUNDER)
    list(APPEND PATCHES ${CMAKE_CURRENT_LIST_DIR}/patch/fix-missing-std-launder.patch008)
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
     AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0
     AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.2)
    list(APPEND PATCHES ${CMAKE_CURRENT_LIST_DIR}/patch/gcc-8.1-cleanup-hpp.patch009)
  endif()

  if(MSVC AND COMPILER_IS_CLANG_CL)
    list(APPEND PATCHES ${CMAKE_CURRENT_LIST_DIR}/patch/clang-cl-compatibility.patch010)
  endif()

  if(Boost_DIR)
    list(APPEND CMAKE_OPTION -DBoost_DIR=${Boost_DIR})
  endif()

  mindquantum_add_pkg(
    tweedledum
    VER 1.1.0
    LIBS tweedledum
    GIT_REPOSITORY ${GIT_URL}
    GIT_TAG ${GIT_TAG}
    MD5 ${MD5}
    PATCHES ${PATCHES}
    CMAKE_OPTION
      ${CMAKE_OPTION} -DEigen3_DIR=${Eigen3_DIR} -Dfmt_DIR=${fmt_DIR} -Dnlohmann_json_DIR=${nlohmann_json_DIR}
      -Dpybind11_DIR=${pybind11_DIR} -DSymEngine_DIR=${SymEngine_DIR}
    TARGET_ALIAS mindquantum::tweedledum tweedledum::tweedledum
    LOCAL_EXTRA_DEFINES ${tweedledum_LOCAL_EXTRA_DEFINES}
    SYSTEM_EXTRA_DEFINES ${tweedledum_SYSTEM_EXTRA_DEFINES})

  foreach(
    _comp
    abcresub
    libabcesop
    bill
    kitty
    rang
    lorina
    percy
    libabcsat
    mockturtle)
    set(_tgt tweedledum::${_comp})
    if(TARGET ${_tgt})
      get_target_property(_aliased ${_tgt} ALIASED_TARGET)
      if(_aliased)
        set(_tgt ${_aliased})
      endif()
      set_target_properties(${_tgt} PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
  endforeach()
endif()
