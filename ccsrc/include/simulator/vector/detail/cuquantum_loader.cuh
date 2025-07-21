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
#ifndef CCSRC_INCLUDE_SIMULATOR_VECTOR_DETAIL_CUQUANTUM_LOADER_CUH_
#define CCSRC_INCLUDE_SIMULATOR_VECTOR_DETAIL_CUQUANTUM_LOADER_CUH_

#include <custatevec.h>  // For type and constant definitions only

namespace mindquantum::sim::vector::detail {

// A struct to hold all required function pointers
struct CuQuantumHandles {
    custatevecStatus_t (*custatevecCreate)(custatevecHandle_t* handle);
    custatevecStatus_t (*custatevecDestroy)(custatevecHandle_t handle);
    const char* (*custatevecGetErrorString)(custatevecStatus_t status);
    custatevecStatus_t (*custatevecApplyMatrix)(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType,
                                                uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType,
                                                custatevecMatrixLayout_t matrixLayout, int32_t adjoint,
                                                const int32_t* targets, uint32_t nTargets, const int32_t* controls,
                                                const int32_t* controlBitValues, uint32_t nControls,
                                                custatevecComputeType_t computeType, void* extraWorkspace,
                                                size_t extraWorkspaceSizeInBytes);
    custatevecStatus_t (*custatevecApplyPauliRotation)(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType,
                                                       uint32_t nIndexBits, double theta,
                                                       const custatevecPauli_t* paulis, const int32_t* targets,
                                                       uint32_t nTargets, const int32_t* controls,
                                                       const int32_t* controlBitValues, uint32_t nControls);
};

class CuQuantumLoader {
 public:
    // Get the singleton instance
    static CuQuantumLoader& GetInstance();

    // Check if cuQuantum is available
    bool IsAvailable() const;

    // Get the struct containing all function pointers
    const CuQuantumHandles& Handles() const;

    // Disable copy and assignment
    CuQuantumLoader(const CuQuantumLoader&) = delete;
    CuQuantumLoader& operator=(const CuQuantumLoader&) = delete;

 private:
    CuQuantumLoader();   // Constructor performs the loading
    ~CuQuantumLoader();  // Destructor releases the library

    void* dl_handle_ = nullptr;  // Handle for the dynamic library
    bool available_ = false;
    CuQuantumHandles handles_{};
};

// A convenient macro for easy API access
#define CUSTATEVEC_API CuQuantumLoader::GetInstance().Handles()

}  // namespace mindquantum::sim::vector::detail

#endif  // CCSRC_INCLUDE_SIMULATOR_VECTOR_DETAIL_CUQUANTUM_LOADER_CUH_
