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

set(VER 9.1.0)

if(DEFINED ENV{CODEHUB_CI} AND "$ENV{CODEHUB_CI}" STREQUAL "1")
  set(REQ_URL "$ENV{CODEHUB_PKG_URL}/fmt-${VER}.tar.gz")
  set(MD5 "21fac48cae8f3b4a5783ae06b443973a")
elseif(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/mirrors_trending/fmt/repository/archive/${VER}.tar.gz")
  set(MD5 "21fac48cae8f3b4a5783ae06b443973a")
else()
  set(REQ_URL "https://github.com/fmtlib/fmt/archive/${VER}.tar.gz")
  set(MD5 "21fac48cae8f3b4a5783ae06b443973a")
endif()

set(CMAKE_OPTION -DFMT_TEST=OFF -DFMT_DOC=OFF -DFMT_SYSTEM_HEADERS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                 -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR})

if(MSVC)
  set(fmt_CXXFLAGS "/Zc:__cplusplus /EHsc /D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS /DWIN32_LEAN_AND_MEAN")
  if(ENABLE_ITERATOR_DEBUG)
    set(fmt_CFLAGS "/D_ITERATOR_DEBUG_LEVEL=${MQ_ITERATOR_DEBUG}")
    set(fmt_CXXFLAGS "${fmt_CXXFLAGS} /D_ITERATOR_DEBUG_LEVEL=${MQ_ITERATOR_DEBUG}")
  endif()
  list(APPEND CMAKE_OPTION -DCMAKE_DEBUG_POSTFIX=d)

  if(CMAKE_MT)
    list(APPEND CMAKE_OPTION -DCMAKE_MT=${CMAKE_MT})
  endif()
endif()

mindquantum_add_pkg(
  fmt
  VER ${VER}
  URL ${REQ_URL}
  MD5 ${MD5}
  CMAKE_PKG_NO_COMPONENTS
  CMAKE_OPTION ${CMAKE_OPTION}
  TARGET_ALIAS 3 mindquantum::fmt fmt::fmt-header-only fmt::fmt)

get_target_property(_aliased_target mindquantum::fmt ALIASED_TARGET)
if(_aliased_target)
  if("${_aliased_target}" STREQUAL "fmt::fmt")
    set(_fmt_use_header_only
        FALSE
        CACHE BOOL "Is mindquantum::fmt using fmt::fmt-header-only")
  else()
    set(_fmt_use_header_only
        TRUE
        CACHE BOOL "Is mindquantum::fmt using fmt::fmt-header-only")
  endif()
endif()
