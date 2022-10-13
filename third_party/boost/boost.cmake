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

set(VER 1.78.0)
string(REPLACE "." "_" _ver "${VER}")

if(ENABLE_GITEE)
  set(REQ_URL "https://udomain.dl.sourceforge.net/project/boost/boost/${VER}/boost_${_ver}.tar.gz")
  set(MD5 "c2f6428ac52b0e5a3c9b2e1d8cc832b5")
else()
  if(MSVC OR "${OS_NAME}" STREQUAL "MinGW")
    set(REQ_URL "https://boostorg.jfrog.io/artifactory/main/release/${VER}/source/boost_${_ver}.zip")
    set(MD5 "e193e5089060ed6ce5145c8eb05e67e3")
  else()
    set(REQ_URL "https://boostorg.jfrog.io/artifactory/main/release/${VER}/source/boost_${_ver}.tar.gz")
    set(MD5 "c2f6428ac52b0e5a3c9b2e1d8cc832b5")
  endif()
endif()

if(MSVC OR "${OS_NAME}" STREQUAL "MinGW")
  set(PRE_CONFIGURE_COMMAND bootstrap.bat)
  if(NOT MSVC)
    list(APPEND PRE_CONFIGURE_COMMAND mingw)
  endif()
else()
  set(PRE_CONFIGURE_COMMAND ./bootstrap.sh)
endif()

set(TARGET_ALIAS mindquantum::boost_headers Boost::headers)

set(LIBS serialization)
set(INSTALL_COMMAND ./b2 --with-headers --prefix=@PREFIX@ --with-serialization)
list(APPEND TARGET_ALIAS mindquantum::boost_serialization Boost::serialization)
if(NOT MQ_CXX_HAS_STD_FILESYSTEM)
  list(APPEND LIBS filesystem system)
  list(APPEND INSTALL_COMMAND --with-system --with-filesystem)
  list(APPEND TARGET_ALIAS mindquantum::boost_system Boost::system)
  list(APPEND TARGET_ALIAS mindquantum::boost_filesystem Boost::filesystem)
endif()
list(APPEND INSTALL_COMMAND variant=debug,release)
if(UNIX)
  list(APPEND INSTALL_COMMAND --layout=tagged)
endif()

if("${OS_NAME}" STREQUAL "MinGW")
  list(APPEND INSTALL_COMMAND toolset=gcc)
elseif(MSYS AND NOT "${OS_NAME}" STREQUAL "MSYS-MSYS")
  message(WARNING "Build of Boost on MSYS2 (${OS_NAME}) is broken. Make sure to install the corresponding "
                  "mingw-w64-*-boost package instead.")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND PRE_CONFIGURE_COMMAND --with-toolset=clang)
  list(APPEND INSTALL_COMMAND toolset=clang)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND PRE_CONFIGURE_COMMAND --with-toolset=gcc)
  list(APPEND INSTALL_COMMAND toolset=gcc)
endif()
list(APPEND INSTALL_COMMAND install)

mindquantum_add_pkg(
  Boost
  LIBS ${LIBS}
  VER ${VER}
  URL ${REQ_URL}
  MD5 ${MD5}
  PRE_CONFIGURE_COMMAND ${PRE_CONFIGURE_COMMAND}
  SKIP_BUILD_STEP
  INSTALL_COMMAND ${INSTALL_COMMAND}
  TARGET_ALIAS ${TARGET_ALIAS})
