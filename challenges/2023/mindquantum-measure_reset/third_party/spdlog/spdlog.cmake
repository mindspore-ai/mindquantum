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

# lint_cmake: -whitespace/indent

set(VER 1.10.0)

if(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/liveever/spdlog/repository/archive/v${VER}.tar.gz")
  set(MD5 "1d99489389f8fdc1915f98f2887f3e8f")
else()
  set(REQ_URL "https://github.com/gabime/spdlog/archive/v${VER}.tar.gz")
  set(MD5 "effea813cd81cfb5588806c5754e14f1")
endif()

set(CMAKE_OPTION
    -DBUILD_TESTING=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
    -DSPDLOG_BUILD_SHARED=OFF
    -DSPDLOG_BUILD_EXAMPLE=OFF
    -DSPDLOG_BUILD_EXAMPLE_HO=OFF
    -DSPDLOG_BUILD_TESTS=OFF
    -DSPDLOG_BUILD_TESTS_HO=OFF
    -DSPDLOG_BUILD_BENCH=OFF
    -DSPDLOG_SANITIZE_ADDRESS=OFF
    -DSPDLOG_BUILD_WARNINGS=OFF
    -DSPDLOG_INSTALL=ON
    -DSPDLOG_TIDY=OFF
    -Dfmt_DIR=${fmt_DIR})

if(MSVC)
  set(spdlog_CXXFLAGS "/Zc:__cplusplus /EHsc /D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS /DWIN32_LEAN_AND_MEAN")
  if(ENABLE_ITERATOR_DEBUG)
    set(spdlog_CFLAGS "/D_ITERATOR_DEBUG_LEVEL=${MQ_ITERATOR_DEBUG}")
    set(spdlog_CXXFLAGS "${spdlog_CXXFLAGS} /D_ITERATOR_DEBUG_LEVEL=${MQ_ITERATOR_DEBUG}")
  endif()

  if(CMAKE_MT)
    list(APPEND CMAKE_OPTION -DCMAKE_MT=${CMAKE_MT})
  endif()
endif()

if(_fmt_use_header_only)
  list(APPEND CMAKE_OPTION -DSPDLOG_FMT_EXTERNAL_HO=ON)
else()
  list(APPEND CMAKE_OPTION -DSPDLOG_FMT_EXTERNAL=ON)
endif()

mindquantum_add_pkg(
  spdlog
  VER ${VER}
  FORCE_EXACT_VERSION
  URL ${REQ_URL}
  MD5 ${MD5}
  CMAKE_OPTION ${CMAKE_OPTION}
  TARGET_ALIAS mindquantum::spdlog spdlog::spdlog)
