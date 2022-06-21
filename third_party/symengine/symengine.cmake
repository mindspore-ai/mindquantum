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

# cmake-lint: disable=C0103

set(VER 0.9.0)

if(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/mirrors/SymEngine/repository/archive/v${VER}.tar.gz")
  set(MD5 "e2cf233dd7afd85f3117384e4fc21624")
else()
  set(REQ_URL "https://github.com/symengine/symengine/archive/v${VER}.tar.gz")
  set(MD5 "72d11e59315b84ff9abdf51c590e986c")
endif()

set(CMAKE_OPTION -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON)

if(MSVC)
  set(SymEngine_CFLAGS "/D_ITERATOR_DEBUG_LEVEL=0")
  set(SymEngine_CXXFLAGS "/D_ITERATOR_DEBUG_LEVEL=0 /Zc:__cplusplus /EHsc")
  list(APPEND CMAKE_OPTION -DINTEGER_CLASS=boostmp -DCMAKE_DEBUG_POSTFIX=d)

  if(Boost_DIRPATH)
    # Boost was locally built, make sure we use that one
    list(APPEND CMAKE_OPTION -DBOOST_ROOT=${Boost_DIRPATH} -DBoost_NO_SYSTEM_PATHS:BOOL=ON)
  endif()
elseif("${OS_NAME}" STREQUAL "MinGW")
  list(APPEND CMAKE_OPTION -DINTEGER_CLASS=boostmp -DCMAKE_DEBUG_POSTFIX=d)

  if(Boost_DIRPATH)
    # Boost was locally built, make sure we use that one
    list(APPEND CMAKE_OPTION -DBOOST_ROOT=${Boost_DIRPATH} -DBoost_NO_SYSTEM_PATHS:BOOL=ON)
  endif()
endif()

set(PATCHES
    ${CMAKE_CURRENT_LIST_DIR}/patch/fix-cmakelists.patch001
    ${CMAKE_CURRENT_LIST_DIR}/patch/remove-msvc-mt-option.patch002
    ${CMAKE_CURRENT_LIST_DIR}/patch/fix-finding-gmp.patch003)

if(gmp_DIR)
  # If gmp_DIR is defined then we can find gmp using the CONFIG method -> patch SymEngine accordingly.
  #
  # Also, this will only happen if we compiled gmp locally since gmp by default does not provide CMake configuration
  # files during its normal installation. In all other case, the FindGMP code from SymEngine should be able to locate
  # gmp successfully.
  list(APPEND CMAKE_OPTION -Dgmp_DIR=${gmp_DIR})
  list(APPEND PATCHES ${CMAKE_CURRENT_LIST_DIR}/patch/find-gmp-using-config-method.patch004)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  list(APPEND PATCHES ${CMAKE_CURRENT_LIST_DIR}/patch/apple-clang-debug-build-fix.patch005)
endif()

mindquantum_add_pkg(
  SymEngine
  VER ${VER}
  URL ${REQ_URL}
  MD5 ${MD5}
  CMAKE_OPTION ${CMAKE_OPTION}
  PATCHES ${PATCHES}
  TARGET_ALIAS mindquantum::symengine symengine)

if(TARGET mindquantum::gmp)
  get_target_property(_link_libraries symengine INTERFACE_LINK_LIBRARIES)

  if(gmp IN_LIST _link_libraries)
    list(REMOVE_ITEM _link_libraries gmp)
    set_target_properties(symengine PROPERTIES INTERFACE_LINK_LIBRARIES "${_link_libraries}")
  endif()

  if(NOT mindquantum::gmp IN_LIST _link_libraries)
    target_link_libraries(symengine INTERFACE mindquantum::gmp)
  endif()
endif()
