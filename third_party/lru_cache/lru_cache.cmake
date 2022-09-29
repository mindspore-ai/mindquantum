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

set(VER 0.0.1)

if(ENABLE_GITEE)
  set(GIT_URL "https://gitee.com/dnguyen/lru_cache.git")
else()
  set(GIT_URL "https://github.com/nitnelave/lru_cache.git")
endif()
set(GIT_TAG "48f9cd9f1713ee63172f2a28af6824bbf8161e5c")

set(PATCHES ${CMAKE_CURRENT_LIST_DIR}/patch/fix-memory-leak-node-deletion-callback.patch001
            ${CMAKE_CURRENT_LIST_DIR}/patch/optional-abseil-dependency.patch002)

mindquantum_add_pkg(
  lru_cache
  VER ${VER}
  GIT_REPOSITORY ${GIT_URL}
  GIT_TAG ${GIT_TAG}
  MD5 "xxxx" # NB: would be required if local server is enabled for downloads
  ONLY_COPY_DIRS lru_cache
  PATCHES ${PATCHES}
  FORCE_LOCAL_PKG
  TARGET_ALIAS mindquantum::lru_cache lru_cache::lru_cache)
if(TARGET lru_cache::lru_cache)
  target_compile_features(lru_cache::lru_cache INTERFACE cxx_std_17)
  if(ENABLE_ABSEIL_CPP)
    target_link_libraries(lru_cache::lru_cache INTERFACE mindquantum::absl_node_hash_set)
    target_compile_definitions(lru_cache::lru_cache INTERFACE LRU_CACHE_HAS_ABSEIL_CPP)
  endif()
endif()
