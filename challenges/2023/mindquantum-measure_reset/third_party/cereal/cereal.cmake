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

set(VER 1.3.2)

if(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/mirrors/cereal/repository/archive/v${VER}.tar.gz")
  set(MD5 "bb0d381719ef49d394b30aa16ec401b3")
else()
  set(REQ_URL "https://github.com/uscilab/cereal/archive/v${VER}.tar.gz")
  set(MD5 "ab6070fadc7c50072ef4153fb1c46a87")
endif()

set(CMAKE_OPTION
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_TESTS=OFF
    -DBUILD_DOC=OFF
    -DSKIP_PORTABILITY_TEST=ON
    -DBUILD_SANDBOX=OFF
    -DCEREAL_INSTALL=ON
    -DWITH_WERROR=OFF
    -DSKIP_PERFORMANCE_COMPARISON=ON)

if(MSVC AND CMAKE_MT)
  list(APPEND CMAKE_OPTION -DCMAKE_MT=${CMAKE_MT})
endif()

mindquantum_add_pkg(
  cereal
  VER ${VER}
  URL ${REQ_URL}
  MD5 ${MD5}
  CMAKE_OPTION ${CMAKE_OPTION}
  TARGET_ALIAS mindquantum::cereal cereal::cereal)
