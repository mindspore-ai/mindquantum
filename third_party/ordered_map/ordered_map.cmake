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

set(VER 1.0.0)

if(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/wenqzheng/ordered-map/repository/archive/v${VER}.tar.gz")
  set(MD5 "711adbfa8d43b2c52e2f309c146c8334")
else()
  set(REQ_URL "https://github.com/Tessil/ordered-map/archive/v${VER}.tar.gz")
  set(MD5 "1776b899e7571ba77ffc127ac42a91ef")
endif()

mindquantum_add_pkg(
  tsl-ordered-map
  VER ${VER}
  URL ${REQ_URL}
  MD5 ${MD5}
  CMAKE_OPTION "-DBUILD_TESTING=OFF"
  TARGET_ALIAS mindquantum::ordered_map tsl::ordered_map)
