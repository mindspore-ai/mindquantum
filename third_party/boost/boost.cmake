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

# NB: Ubuntu 20.04 LTS has 1.71.0, Debian Buster 1.67.0
set(VER 1.67.0) # Version provided by Ubuntu 20.04 LTS
if(MQ_FORCE_LOCAL_PKGS OR MQ_BOOST_FORCE_LOCAL)
  set(VER 1.78.0)
endif()

set(_version_for_download 1.78.0)
string(REPLACE "." "_" _ver "${_version_for_download}")

if(MSVC OR "${OS_NAME}" STREQUAL "MinGW")
  set(REQ_URL "https://boostorg.jfrog.io/artifactory/main/release/${_version_for_download}/source/boost_${_ver}.zip")
  set(MD5 "e193e5089060ed6ce5145c8eb05e67e3")

  if(MSVC)
    set(PRE_CONFIGURE_COMMAND bootstrap.bat)
  else()
    set(PRE_CONFIGURE_COMMAND bootstrap.bat mingw)
  endif()
else()
  set(REQ_URL "https://boostorg.jfrog.io/artifactory/main/release/${_version_for_download}/source/boost_${_ver}.tar.gz")
  set(MD5 "c2f6428ac52b0e5a3c9b2e1d8cc832b5")

  set(PRE_CONFIGURE_COMMAND ./bootstrap.sh)
endif()

set(TARGET_ALIAS mindquantum::boost_headers Boost::headers)

set(LIBS CMAKE_PKG_NO_COMPONENTS)
set(INSTALL_COMMAND ./b2 --with-headers --prefix=@PREFIX@)
if(NOT MQ_HAS_STD_FILESYSTEM)
  set(LIBS LIBS filesystem system)
  list(APPEND INSTALL_COMMAND --with-system --with-filesystem)
  list(APPEND TARGET_ALIAS mindquantum::boost_system Boost::system)
  list(APPEND TARGET_ALIAS mindquantum::boost_filesystem Boost::filesystem)
  list(APPEND INSTALL_COMMAND variant=debug,release)
  if(UNIX)
    list(APPEND INSTALL_COMMAND --layout=tagged)
  endif()
endif()
if("${OS_NAME}" STREQUAL "MinGW")
  list(APPEND INSTALL_COMMAND toolset=gcc)
endif()
list(APPEND INSTALL_COMMAND install)

mindquantum_add_pkg(
  Boost ${LIBS}
  VER ${VER}
  URL ${REQ_URL}
  MD5 ${MD5}
  PRE_CONFIGURE_COMMAND ${PRE_CONFIGURE_COMMAND}
  SKIP_BUILD_STEP
  INSTALL_COMMAND ${INSTALL_COMMAND}
  TARGET_ALIAS ${TARGET_ALIAS})
