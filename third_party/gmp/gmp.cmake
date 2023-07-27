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

set(VER 6.2.0)

if(DEFINED ENV{CODEHUB_CI} AND "$ENV{CODEHUB_CI}" STREQUAL "1")
  set(REQ_URL "$ENV{CODEHUB_PKG_URL}/gmp-${VER}.tar.xz")
  set(MD5 "a325e3f09e6d91e62101e59f9bda3ec1")
else()
  set(REQ_URL "https://gmplib.org/download/gmp/gmp-${VER}.tar.xz")
  set(MD5 "a325e3f09e6d91e62101e59f9bda3ec1")
endif()

set(gmp_CFLAGS "-fPIC -O3 -D_FORTIFY_SOURCE=2")
set(gmp_CXXFLAGS "-fPIC -O3 -D_FORTIFY_SOURCE=2")

if(MSYS)
  message(WARNING "Build of gmp on MSYS2 is broken. Make sure to install the gmp-devel (MSYS) or mingw-w64-*-gmp "
                  "packages instead.")
elseif(CYGWIN)
  message(WARNING "Build of gmp on Cygwin is broken. Make sure to install the libgmp-devel package instead.")
endif()

mindquantum_add_pkg(
  gmp
  LIBS gmp gmpxx
  VER ${VER}
  URL ${REQ_URL}
  MD5 ${MD5}
  GEN_CMAKE_CONFIG
  BUILD_DEPENDENCIES "m4"
  CONFIGURE_COMMAND ./configure --enable-cxx --enable-shared=no --with-pic
  TARGET_ALIAS mindquantum::gmp gmp::gmp)
