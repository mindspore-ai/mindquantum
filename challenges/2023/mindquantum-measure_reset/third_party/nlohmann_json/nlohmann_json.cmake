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

set(CMAKE_OPTION -DBUILD_TESTING=OFF -DJSON_MultipleHeaders=ON -DJSON_BuildTests=OFF
                 -DCMAKE_POSITION_INDEPENDENT_CODE=ON)

if(MSVC)
  set(nlohmann_json_CXXFLAGS
      "/Zc:__cplusplus /EHsc /D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS /DWIN32_LEAN_AND_MEAN")
  if(ENABLE_ITERATOR_DEBUG)
    set(nlohmann_json_CFLAGS "/D_ITERATOR_DEBUG_LEVEL=${MQ_ITERATOR_DEBUG}")
    set(nlohmann_json_CXXFLAGS "${nlohmann_json_CXXFLAGS} /D_ITERATOR_DEBUG_LEVEL=${MQ_ITERATOR_DEBUG}")
  endif()
  list(APPEND CMAKE_OPTION -DCMAKE_DEBUG_POSTFIX=d)
  if(CMAKE_MT)
    list(APPEND CMAKE_OPTION -DCMAKE_MT=${CMAKE_MT})
  endif()
else()
  set(nlohmann_json_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
  set(nlohmann_json_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
endif()

set(VER 3.11.2)

if(DEFINED ENV{CODEHUB_CI} AND "$ENV{CODEHUB_CI}" STREQUAL "1")
  set(REQ_URL "$ENV{CODEHUB_PKG_URL}/json-${VER}.tar.gz")
  set(MD5 "e8d56bc54621037842ee9f0aeae27746")
elseif(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/mirrors/JSON-for-Modern-CPP/repository/archive/v${VER}.tar.gz")
  set(MD5 "03c61931b62a1095034dc6a0b5118260")
else()
  set(REQ_URL "https://github.com/nlohmann/json/archive/v${VER}.tar.gz")
  set(MD5 "e8d56bc54621037842ee9f0aeae27746")
endif()

mindquantum_add_pkg(
  nlohmann_json
  VER ${VER}
  URL ${REQ_URL}
  MD5 ${MD5}
  CMAKE_PKG_NO_COMPONENTS
  CMAKE_OPTION ${CMAKE_OPTION}
  TARGET_ALIAS mindquantum::json nlohmann_json::nlohmann_json)
