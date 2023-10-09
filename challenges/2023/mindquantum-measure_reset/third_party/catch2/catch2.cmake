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

# cmake-lint: disable=C0103

set(VER 3.1.0)

if(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/zhaozonggang100/Catch/repository/archive/v${VER}.tar.gz")
  set(MD5 "a553afdcef5becdb2550c671a57962d5")
else()
  set(REQ_URL "https://github.com/catchorg/Catch2/archive/refs/tags/v${VER}.tar.gz")
  set(MD5 "730ddfe3a57b5316f790b0347c217651")
endif()

set(CMAKE_OPTION
    -DCATCH_BUILD_TESTING=OFF
    -DCATCH_INSTALL_DOCS=OFF
    -DCATCH_INSTALL_EXTRAS=ON
    -DCATCH_CONFIG_WCHAR=OFF
    -DCATCH_CONFIG_ENABLE_ALL_STRINGMAKERS=ON
    -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON)

set(Catch2WithMain_LOCAL_EXTRA_DEFINES)
set(Catch2WithMain_SYSTEM_EXTRA_DEFINES)
set(Catch2_LOCAL_EXTRA_DEFINES)
set(Catch2_SYSTEM_EXTRA_DEFINES TARGET Catch2::Catch2 CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS)

if(MSVC)
  set(Catch2_CXXFLAGS "/Zc:__cplusplus /EHsc /D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS /DWIN32_LEAN_AND_MEAN")
  if(ENABLE_ITERATOR_DEBUG)
    set(Catch2_CFLAGS "/D_ITERATOR_DEBUG_LEVEL=${MQ_ITERATOR_DEBUG}")
    set(Catch2_CXXFLAGS "${Catch2_CXXFLAGS} /D_ITERATOR_DEBUG_LEVEL=${MQ_ITERATOR_DEBUG}")
  endif()

  if(CMAKE_MT)
    list(APPEND CMAKE_OPTION -DCMAKE_MT=${CMAKE_MT})
  endif()
endif()

if(POSIX_C_SOURCE)
  list(APPEND Catch2WithMain_LOCAL_EXTRA_DEFINES TARGET Catch2::Catch2WithMain "_POSIX_C_SOURCE=${POSIX_C_SOURCE}")
  list(APPEND Catch2WithMain_SYSTEM_EXTRA_DEFINES TARGET Catch2::Catch2WithMain "_POSIX_C_SOURCE=${POSIX_C_SOURCE}")
  list(APPEND Catch2_LOCAL_EXTRA_DEFINES TARGET Catch2::Catch2 "_POSIX_C_SOURCE=${POSIX_C_SOURCE}")
  list(APPEND Catch2_SYSTEM_EXTRA_DEFINES "_POSIX_C_SOURCE=${POSIX_C_SOURCE}")
  set(Catch2_CFLAGS "-D_POSIX_C_SOURCE=${POSIX_C_SOURCE}")
  set(Catch2_CXXFLAGS "-D_POSIX_C_SOURCE=${POSIX_C_SOURCE}")
endif()

# cmake-lint: disable=E1122
mindquantum_add_pkg(
  Catch2
  VER ${VER}
  URL ${REQ_URL}
  MD5 ${MD5}
  CMAKE_PKG_NO_COMPONENTS
  CMAKE_OPTION ${CMAKE_OPTION}
  TARGET_ALIAS mindquantum::catch2 Catch2::Catch2
  TARGET_ALIAS mindquantum::catch2_main Catch2::Catch2WithMain
  SKIP_IN_INSTALL_CONFIG
  LOCAL_EXTRA_DEFINES ${Catch2_LOCAL_EXTRA_DEFINES} ${Catch2WithMain_LOCAL_EXTRA_DEFINES}
  SYSTEM_EXTRA_DEFINES ${Catch2_SYSTEM_EXTRA_DEFINES} ${Catch2WithMain_SYSTEM_EXTRA_DEFINES})
