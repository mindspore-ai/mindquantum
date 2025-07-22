/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dlfcn.h>  // For dlopen, dlsym, dlclose on Linux/macOS

#include <iostream>

#include "simulator/vector/detail/cuquantum_loader.cuh"

namespace mindquantum::sim::vector::detail {

// Helper macro to simplify loading functions and checking for errors
#define LOAD_CUQUANTUM_FUNC(name)                                                                                      \
    handles_.name = reinterpret_cast<decltype(handles_.name)>(dlsym(dl_handle_, #name));                               \
    if (!handles_.name) {                                                                                              \
        std::cerr << "MindQuantum Warning: Failed to load cuQuantum function: " << #name << std::endl;                 \
        available_ = false;                                                                                            \
        return;                                                                                                        \
    }

CuQuantumLoader& CuQuantumLoader::GetInstance() {
    static CuQuantumLoader instance;
    return instance;
}

CuQuantumLoader::CuQuantumLoader() {
    // Try to open the cuQuantum library. RTLD_LAZY defers symbol resolution until first use.
    dl_handle_ = dlopen("libcustatevec.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!dl_handle_) {
        // This is not a fatal error. It simply means the cuQuantum backend won't be available.
        return;
    }

    available_ = true;

    // Load all the required function pointers
    LOAD_CUQUANTUM_FUNC(custatevecCreate);
    LOAD_CUQUANTUM_FUNC(custatevecDestroy);
    LOAD_CUQUANTUM_FUNC(custatevecGetErrorString);
    LOAD_CUQUANTUM_FUNC(custatevecApplyMatrix);
    LOAD_CUQUANTUM_FUNC(custatevecApplyPauliRotation);
}

CuQuantumLoader::~CuQuantumLoader() {
    if (dl_handle_) {
        dlclose(dl_handle_);
    }
}

bool CuQuantumLoader::IsAvailable() const {
    return available_;
}

const CuQuantumHandles& CuQuantumLoader::Handles() const {
    // It is the caller's responsibility to check IsAvailable() before calling this.
    return handles_;
}
}  // namespace mindquantum::sim::vector::detail
