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

set(VER 20220623)
set(REV 0)

if(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/mirrors/abseil-cpp/repository/archive/${VER}.${REV}.tar.gz")
  set(MD5 "955b6faedf32ec2ce1b7725561d15618")
else()
  set(REQ_URL "https://github.com/abseil/abseil-cpp/archive/${VER}.${REV}.tar.gz")
  set(MD5 "955b6faedf32ec2ce1b7725561d15618")
endif()

set(CMAKE_OPTION -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DABSL_PROPAGATE_CXX_STD=ON
                 -DABSL_ENABLE_INSTALL=ON)

mindquantum_add_pkg(
  absl
  VER ${VER}
  FORCE_EXACT_VERSION
  LIBS node_hash_set
  URL ${REQ_URL}
  MD5 ${MD5}
  CMAKE_OPTION ${CMAKE_OPTION}
  TARGET_ALIAS mindquantum::absl_node_hash_set absl::node_hash_set)
