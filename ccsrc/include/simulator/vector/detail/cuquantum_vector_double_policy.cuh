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
#ifndef INCLUDE_VECTOR_DETAIL_CUQUANTUM_VECTOR_DOUBLE_POLICY_CUH
#define INCLUDE_VECTOR_DETAIL_CUQUANTUM_VECTOR_DOUBLE_POLICY_CUH

#include <custatevec.h>  // Include cuStateVec library
#include <string>

#include "simulator/vector/detail/gpu_vector_policy.cuh"

namespace mindquantum::sim::vector::detail {

// Structure defining the cuQuantum vector policy for double-precision floating-point numbers.
// Inherits from GPUVectorPolicyBase to reuse common functionalities.
struct CuQuantumVectorPolicyDouble : public GPUVectorPolicyBase<CuQuantumVectorPolicyDouble, double> {
    using Base = GPUVectorPolicyBase<CuQuantumVectorPolicyDouble, double>;
    static custatevecHandle_t handle_;  // cuStateVec handle

    // Initialize cuStateVec handle
    static void Init();
    // Destroy cuStateVec handle
    static void Destroy();

    // Override gate application methods here using cuQuantum.
    static void ApplyH(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyRX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRzz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);

    // Gates to implement
    static void ApplyY(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyZ(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplySGate(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplySdag(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyT(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyTdag(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplySX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplySXdag(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyGP(qs_data_p_t* qs_p, qbit_t obj_qubit, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);

    // Rotation gates (implement non-diff versions)
    static void ApplyRY(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRZ(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyPS(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRxx(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRyy(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRxy(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRxz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRyz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);

    // Two-qubit gates
    static void ApplySWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyISWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, bool daggered, index_t dim);
    static void ApplyGivens(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                            bool diff = false);
    static void ApplySWAPalpha(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                               bool diff = false);

 private:
    // Helper for applying 4x4 matrix (double)
    static void apply_two_qubit_matrix(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                       const std::complex<double>* matrix, index_t dim, const std::string& gate_name);
};

}  // namespace mindquantum::sim::vector::detail

#endif  // INCLUDE_VECTOR_DETAIL_CUQUANTUM_VECTOR_DOUBLE_POLICY_CUH
