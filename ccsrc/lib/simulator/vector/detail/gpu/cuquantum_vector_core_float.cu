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

#include <cmath>

#include <complex>
#include <stdexcept>
#include <vector>

#include "config/openmp.h"
#include "simulator/utils.h"
#include "simulator/vector/detail/cuquantum_loader.cuh"
#include "simulator/vector/detail/cuquantum_vector_float_policy.cuh"

namespace mindquantum::sim::vector::detail {

namespace {
const std::complex<float> h_matrix[] = {{M_SQRT1_2, 0.0f}, {M_SQRT1_2, 0.0f}, {M_SQRT1_2, 0.0f}, {-M_SQRT1_2, 0.0f}};
const std::complex<float> x_matrix[] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}};
const std::complex<float> y_matrix[] = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {0.0f, -1.0f},
    {0.0f, 0.0f}};  // Col Major: {{0,0}, {0,-1}}, {{0,1}, {0,0}} -> {(0,0), (0,1)}, {(0,-1), (0,0)}
const std::complex<float> z_matrix[] = {{1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {-1.0f, 0.0f}};
const std::complex<float> s_matrix[] = {{1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 1.0f}};
const std::complex<float> sdag_matrix[] = {{1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, -1.0f}};
const float t_angle_factor = M_SQRT1_2;  // Reusable factor for T gates
const std::complex<float> t_matrix[] = {{1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {t_angle_factor, t_angle_factor}};
const std::complex<float> tdag_matrix[] = {{1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {t_angle_factor, -t_angle_factor}};
const std::complex<float> sx_matrix[] = {{0.5f, 0.5f},
                                         {0.5f, -0.5f},
                                         {0.5f, -0.5f},
                                         {0.5f, 0.5f}};  // Col Major: {{0.5+0.5i, 0.5-0.5i}, {0.5-0.5i, 0.5+0.5i}} ->
                                                         // {(0.5,0.5), (0.5,-0.5)}, {(0.5,-0.5), (0.5,0.5)}
const std::complex<float> sxdag_matrix[] = {{0.5f, -0.5f},
                                            {0.5f, 0.5f},
                                            {0.5f, 0.5f},
                                            {0.5f, -0.5f}};  // Col Major: {{0.5-0.5i, 0.5+0.5i}, {0.5+0.5i, 0.5-0.5i}}
                                                             // -> {(0.5,-0.5), (0.5,0.5)}, {(0.5,0.5), (0.5,-0.5)}
const std::complex<float> swap_matrix[] = {                  // Col Major for {{1,0,0,0},{0,0,1,0},{0,1,0,0},{0,0,0,1}}
    {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f},
    {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}};
}  // namespace

custatevecHandle_t CuQuantumVectorPolicyFloat::handle_ = nullptr;

inline void checkCuStateVecError(custatevecStatus_t status, const std::string& func_name) {
    if (status != CUSTATEVEC_STATUS_SUCCESS) {
        const char* err_str = CUSTATEVEC_API.custatevecGetErrorString(status);
        throw std::runtime_error("cuStateVec Error in function " + func_name + ": "
                                 + (err_str ? err_str : "Unknown error"));
    }
}

void CuQuantumVectorPolicyFloat::Init() {
    if (!CuQuantumLoader::GetInstance().IsAvailable()) {
        throw std::runtime_error("cuQuantum backend is not available. Please ensure cuQuantum is installed correctly.");
    }
    if (handle_ == nullptr) {
        checkCuStateVecError(CUSTATEVEC_API.custatevecCreate(&handle_), "custatevecCreate");
    }
}

void CuQuantumVectorPolicyFloat::Destroy() {
    if (handle_ != nullptr) {
        checkCuStateVecError(CUSTATEVEC_API.custatevecDestroy(handle_), "custatevecDestroy");
        handle_ = nullptr;
    }
}

void CuQuantumVectorPolicyFloat::ApplyH(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }

    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));
    auto sv_data_type = CUDA_C_32F;
    void* sv_ptr = reinterpret_cast<void*>(qs);
    auto matrix_data_type = CUDA_C_32F;
    auto layout = CUSTATEVEC_MATRIX_LAYOUT_COL;
    int adjoint = 0;

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError(
        CUSTATEVEC_API.custatevecApplyMatrix(handle_, sv_ptr, sv_data_type, n_qubits, h_matrix, matrix_data_type,
                                             layout, adjoint, target_bits.data(), target_bits.size(), ctrl_ptr, nullptr,
                                             control_bits.size(), CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
        "custatevecApplyMatrix (H)");
}

void CuQuantumVectorPolicyFloat::ApplyX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }

    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));  // Calculate n_qubits from dim
    auto sv_data_type = CUDA_C_32F;
    void* sv_ptr = reinterpret_cast<void*>(qs);
    auto matrix_data_type = CUDA_C_32F;
    auto layout = CUSTATEVEC_MATRIX_LAYOUT_COL;
    int adjoint = 0;

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError(
        CUSTATEVEC_API.custatevecApplyMatrix(handle_, sv_ptr, sv_data_type, n_qubits, x_matrix, matrix_data_type,
                                             layout, adjoint, target_bits.data(), target_bits.size(), ctrl_ptr, nullptr,
                                             control_bits.size(), CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
        "custatevecApplyMatrix (X)");
}

void CuQuantumVectorPolicyFloat::ApplyRX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    if (diff) {
        Base::ApplyRX(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    if (objs.size() != 1) {
        throw std::invalid_argument("RX gate requires exactly one target qubit.");
    }
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }

    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));
    auto sv_data_type = CUDA_C_32F;
    void* sv_ptr = reinterpret_cast<void*>(qs);

    // Angle transformation: RX(θ) = exp(-i θ/2 X), but custatevecApplyPauliRotation implements exp(i angle P).
    // So, we need angle = -θ/2.
    float angle_for_pauli_rot = -static_cast<float>(val) / 2.0f;

    std::vector<int32_t> target_bits = {static_cast<int32_t>(objs[0])};
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_X};
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError(CUSTATEVEC_API.custatevecApplyPauliRotation(
                             handle_, sv_ptr, sv_data_type, n_qubits, angle_for_pauli_rot, paulis, target_bits.data(),
                             target_bits.size(), ctrl_ptr, nullptr, control_bits.size()),
                         "custatevecApplyPauliRotation (RX)");
}

void CuQuantumVectorPolicyFloat::ApplyRzz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                          index_t dim, bool diff) {
    if (diff) {
        Base::ApplyRzz(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    if (objs.size() != 2) {
        throw std::invalid_argument("Rzz gate requires exactly two target qubits.");
    }
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }

    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));
    auto sv_data_type = CUDA_C_32F;
    void* sv_ptr = reinterpret_cast<void*>(qs);

    // Angle transformation: Rzz(θ) = exp(-i θ/2 ZZ), but custatevecApplyPauliRotation implements exp(i angle P).
    // So, we need angle = -θ/2.
    float angle_for_pauli_rot = -static_cast<float>(val) / 2.0f;

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_Z, CUSTATEVEC_PAULI_Z};

    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError(CUSTATEVEC_API.custatevecApplyPauliRotation(
                             handle_, sv_ptr, sv_data_type, n_qubits, angle_for_pauli_rot, paulis, target_bits.data(),
                             target_bits.size(), ctrl_ptr, nullptr, control_bits.size()),
                         "custatevecApplyPauliRotation (Rzz)");
}

#define APPLY_STD_SINGLE_QUBIT_MATRIX(GATE_NAME, MATRIX)                                                               \
    void CuQuantumVectorPolicyFloat::Apply##GATE_NAME(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,    \
                                                      index_t dim) {                                                   \
        Init();                                                                                                        \
        auto& qs = *qs_p;                                                                                              \
        if (qs == nullptr) {                                                                                           \
            qs = Base::InitState(dim);                                                                                 \
        }                                                                                                              \
        uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));                                \
        auto sv_data_type = CUDA_C_32F;                                                                                \
        void* sv_ptr = reinterpret_cast<void*>(qs);                                                                    \
        auto matrix_data_type = CUDA_C_32F;                                                                            \
        auto layout = CUSTATEVEC_MATRIX_LAYOUT_COL;                                                                    \
        int adjoint = 0;                                                                                               \
        std::vector<int32_t> target_bits(objs.begin(), objs.end());                                                    \
        std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());                                                 \
        const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();                                \
        checkCuStateVecError(CUSTATEVEC_API.custatevecApplyMatrix(                                                     \
                                 handle_, sv_ptr, sv_data_type, n_qubits, MATRIX, matrix_data_type, layout, adjoint,   \
                                 target_bits.data(), target_bits.size(), ctrl_ptr, nullptr, control_bits.size(),       \
                                 CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),                                              \
                             "custatevecApplyMatrix (" #GATE_NAME ")");                                                \
    }

APPLY_STD_SINGLE_QUBIT_MATRIX(Y, y_matrix)
APPLY_STD_SINGLE_QUBIT_MATRIX(Z, z_matrix)
APPLY_STD_SINGLE_QUBIT_MATRIX(SGate, s_matrix)
APPLY_STD_SINGLE_QUBIT_MATRIX(Sdag, sdag_matrix)
APPLY_STD_SINGLE_QUBIT_MATRIX(T, t_matrix)
APPLY_STD_SINGLE_QUBIT_MATRIX(Tdag, tdag_matrix)
APPLY_STD_SINGLE_QUBIT_MATRIX(SX, sx_matrix)
APPLY_STD_SINGLE_QUBIT_MATRIX(SXdag, sxdag_matrix)

void CuQuantumVectorPolicyFloat::ApplyRY(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    if (diff) {
        Base::ApplyRY(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    if (objs.size() != 1) {
        throw std::invalid_argument("RY gate requires exactly one target qubit.");
    }
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }
    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));
    auto sv_data_type = CUDA_C_32F;
    void* sv_ptr = reinterpret_cast<void*>(qs);

    // Angle transformation: RY(θ) = exp(-i θ/2 Y), angle = -θ/2.
    float angle_for_pauli_rot = -static_cast<float>(val) / 2.0f;

    std::vector<int32_t> target_bits = {static_cast<int32_t>(objs[0])};
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_Y};
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError(CUSTATEVEC_API.custatevecApplyPauliRotation(
                             handle_, sv_ptr, sv_data_type, n_qubits, angle_for_pauli_rot, paulis, target_bits.data(),
                             target_bits.size(), ctrl_ptr, nullptr, control_bits.size()),
                         "custatevecApplyPauliRotation (RY)");
}

void CuQuantumVectorPolicyFloat::ApplyRZ(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    if (diff) {
        Base::ApplyRZ(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    if (objs.size() != 1) {
        throw std::invalid_argument("RZ gate requires exactly one target qubit.");
    }
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }
    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));
    auto sv_data_type = CUDA_C_32F;
    void* sv_ptr = reinterpret_cast<void*>(qs);

    // Angle transformation: RZ(θ) = exp(-i θ/2 Z), angle = -θ/2.
    float angle_for_pauli_rot = -static_cast<float>(val) / 2.0f;

    std::vector<int32_t> target_bits = {static_cast<int32_t>(objs[0])};
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_Z};
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError(CUSTATEVEC_API.custatevecApplyPauliRotation(
                             handle_, sv_ptr, sv_data_type, n_qubits, angle_for_pauli_rot, paulis, target_bits.data(),
                             target_bits.size(), ctrl_ptr, nullptr, control_bits.size()),
                         "custatevecApplyPauliRotation (RZ)");
}

void CuQuantumVectorPolicyFloat::ApplyPS(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    if (diff) {
        Base::ApplyPS(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }
    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));
    auto sv_data_type = CUDA_C_32F;
    void* sv_ptr = reinterpret_cast<void*>(qs);
    auto matrix_data_type = CUDA_C_32F;
    auto layout = CUSTATEVEC_MATRIX_LAYOUT_COL;
    int adjoint = 0;

    float angle = static_cast<float>(val);
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);
    const std::complex<float> ps_matrix[] = {// Col Major: {{1, 0}, {0, c+is}} -> {1, 0}, {0, (c,s)}
                                             {1.0f, 0.0f},
                                             {0.0f, 0.0f},
                                             {0.0f, 0.0f},
                                             {cos_val, sin_val}};

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError(
        CUSTATEVEC_API.custatevecApplyMatrix(handle_, sv_ptr, sv_data_type, n_qubits, ps_matrix, matrix_data_type,
                                             layout, adjoint, target_bits.data(), target_bits.size(), ctrl_ptr, nullptr,
                                             control_bits.size(), CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
        "custatevecApplyMatrix (PS)");
}

void CuQuantumVectorPolicyFloat::ApplyGP(qs_data_p_t* qs_p, qbit_t obj_qubit, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    if (diff) {
        Base::ApplyGP(qs_p, obj_qubit, ctrls, val, dim, diff);
        return;
    }
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }
    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));
    auto sv_data_type = CUDA_C_32F;
    void* sv_ptr = reinterpret_cast<void*>(qs);
    auto matrix_data_type = CUDA_C_32F;
    auto layout = CUSTATEVEC_MATRIX_LAYOUT_COL;
    int adjoint = 0;

    float angle = static_cast<float>(val);
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);
    const std::complex<float> gp_matrix[] = {{cos_val, -sin_val}, {0.0f, 0.0f}, {0.0f, 0.0f}, {cos_val, -sin_val}};

    std::vector<int32_t> target_bits = {static_cast<int32_t>(obj_qubit)};
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError(
        CUSTATEVEC_API.custatevecApplyMatrix(handle_, sv_ptr, sv_data_type, n_qubits, gp_matrix, matrix_data_type,
                                             layout, adjoint, target_bits.data(), target_bits.size(), ctrl_ptr, nullptr,
                                             control_bits.size(), CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
        "custatevecApplyMatrix (GP)");
}

void CuQuantumVectorPolicyFloat::ApplyRxx(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                          index_t dim, bool diff) {
    if (diff) {
        Base::ApplyRxx(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    if (objs.size() != 2) {
        throw std::invalid_argument("Rxx gate requires exactly two target qubits.");
    }
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }

    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));
    auto sv_data_type = CUDA_C_32F;
    void* sv_ptr = reinterpret_cast<void*>(qs);

    float angle_for_pauli_rot = -static_cast<float>(val) / 2.0f;  // exp(-i theta/2 XX) -> exp(i angle P)

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_X, CUSTATEVEC_PAULI_X};
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError(CUSTATEVEC_API.custatevecApplyPauliRotation(
                             handle_, sv_ptr, sv_data_type, n_qubits, angle_for_pauli_rot, paulis, target_bits.data(),
                             target_bits.size(), ctrl_ptr, nullptr, control_bits.size()),
                         "custatevecApplyPauliRotation (Rxx)");
}

void CuQuantumVectorPolicyFloat::ApplyRyy(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                          index_t dim, bool diff) {
    if (diff) {
        Base::ApplyRyy(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    if (objs.size() != 2) {
        throw std::invalid_argument("Ryy gate requires exactly two target qubits.");
    }
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }

    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));
    auto sv_data_type = CUDA_C_32F;
    void* sv_ptr = reinterpret_cast<void*>(qs);

    float angle_for_pauli_rot = -static_cast<float>(val) / 2.0f;  // exp(-i theta/2 YY) -> exp(i angle P)

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_Y, CUSTATEVEC_PAULI_Y};
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError(CUSTATEVEC_API.custatevecApplyPauliRotation(
                             handle_, sv_ptr, sv_data_type, n_qubits, angle_for_pauli_rot, paulis, target_bits.data(),
                             target_bits.size(), ctrl_ptr, nullptr, control_bits.size()),
                         "custatevecApplyPauliRotation (Ryy)");
}

void CuQuantumVectorPolicyFloat::apply_two_qubit_matrix(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                        const std::complex<float>* matrix, index_t dim,
                                                        const std::string& gate_name) {
    if (objs.size() != 2) {
        throw std::invalid_argument(gate_name + " gate requires exactly two target qubits.");
    }
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }
    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));
    auto sv_data_type = CUDA_C_32F;
    void* sv_ptr = reinterpret_cast<void*>(qs);
    auto matrix_data_type = CUDA_C_32F;
    auto layout = CUSTATEVEC_MATRIX_LAYOUT_COL;
    int adjoint = 0;

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError(
        CUSTATEVEC_API.custatevecApplyMatrix(handle_, sv_ptr, sv_data_type, n_qubits, matrix, matrix_data_type, layout,
                                             adjoint, target_bits.data(), target_bits.size(), ctrl_ptr, nullptr,
                                             control_bits.size(), CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
        "custatevecApplyMatrix (" + gate_name + ")");
}

void CuQuantumVectorPolicyFloat::ApplySWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    apply_two_qubit_matrix(qs_p, objs, ctrls, swap_matrix, dim, "SWAP");
}

void CuQuantumVectorPolicyFloat::ApplyISWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, bool daggered,
                                            index_t dim) {
    float im_val = daggered ? -1.0f : 1.0f;
    // Col Major: {{1,0,0,0}, {0,0,i,0}, {0,i,0,0}, {0,0,0,1}}
    const std::complex<float> iswap_matrix[] = {{1.0f, 0.0f}, {0.0f, 0.0f},   {0.0f, 0.0f},   {0.0f, 0.0f},
                                                {0.0f, 0.0f}, {0.0f, 0.0f},   {0.0f, im_val}, {0.0f, 0.0f},
                                                {0.0f, 0.0f}, {0.0f, im_val}, {0.0f, 0.0f},   {0.0f, 0.0f},
                                                {0.0f, 0.0f}, {0.0f, 0.0f},   {0.0f, 0.0f},   {1.0f, 0.0f}};
    apply_two_qubit_matrix(qs_p, objs, ctrls, iswap_matrix, dim, daggered ? "ISWAPdg" : "ISWAP");
}

void CuQuantumVectorPolicyFloat::ApplyRxy(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                          index_t dim, bool diff) {
    if (diff) {
        Base::ApplyRxy(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    float angle = static_cast<float>(val);
    float c = cosf(angle / 2.0f);
    float s = sinf(angle / 2.0f);
    // Col Major for {{c,0,0,-s},{0,c,s,0},{0,-s,c,0},{s,0,0,c}}
    const std::complex<float> rxy_matrix[] = {
        {c, 0.0f},    {0.0f, 0.0f}, {0.0f, 0.0f}, {s, 0.0f},    {0.0f, 0.0f}, {c, 0.0f},    {s, 0.0f},    {0.0f, 0.0f},
        {0.0f, 0.0f}, {-s, 0.0f},   {c, 0.0f},    {0.0f, 0.0f}, {-s, 0.0f},   {0.0f, 0.0f}, {0.0f, 0.0f}, {c, 0.0f}};
    apply_two_qubit_matrix(qs_p, objs, ctrls, rxy_matrix, dim, "Rxy");
}

void CuQuantumVectorPolicyFloat::ApplyRxz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                          index_t dim, bool diff) {
    if (diff) {
        Base::ApplyRxz(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    float angle = static_cast<float>(val);
    float c = cosf(angle / 2.0f);
    float s = sinf(angle / 2.0f);
    // Col Major for {{c,-is,0,0},{-is,c,0,0},{0,0,c,is},{0,0,is,c}}
    const std::complex<float> rxz_matrix[] = {
        {c, 0.0f},    {0.0f, -s},   {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, -s},   {c, 0.0f},    {0.0f, 0.0f}, {0.0f, 0.0f},
        {0.0f, 0.0f}, {0.0f, 0.0f}, {c, 0.0f},    {0.0f, s},    {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, s},    {c, 0.0f}};
    apply_two_qubit_matrix(qs_p, objs, ctrls, rxz_matrix, dim, "Rxz");
}

void CuQuantumVectorPolicyFloat::ApplyRyz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                          index_t dim, bool diff) {
    if (diff) {
        Base::ApplyRyz(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    float angle = static_cast<float>(val);
    float c = cosf(angle / 2.0f);
    float s = sinf(angle / 2.0f);
    const std::complex<float> ryz_matrix[] = {
        {c, 0.0f},    {s, 0.0f},    {0.0f, 0.0f}, {0.0f, 0.0f}, {-s, 0.0f},   {c, 0.0f},    {0.0f, 0.0f}, {0.0f, 0.0f},
        {0.0f, 0.0f}, {0.0f, 0.0f}, {c, 0.0f},    {-s, 0.0f},   {0.0f, 0.0f}, {0.0f, 0.0f}, {s, 0.0f},    {c, 0.0f}};
    apply_two_qubit_matrix(qs_p, objs, ctrls, ryz_matrix, dim, "Ryz");
}

void CuQuantumVectorPolicyFloat::ApplyGivens(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                             calc_type val, index_t dim, bool diff) {
    if (diff) {
        Base::ApplyGivens(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    float angle = static_cast<float>(val);
    float c = cosf(angle);
    float s = sinf(angle);
    // Col Major for {{1,0,0,0}, {0,c,-s,0}, {0,s,c,0}, {0,0,0,1}}
    const std::complex<float> givens_matrix[] = {
        {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {c, 0.0f},    {s, 0.0f},    {0.0f, 0.0f},
        {0.0f, 0.0f}, {-s, 0.0f},   {c, 0.0f},    {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}};
    apply_two_qubit_matrix(qs_p, objs, ctrls, givens_matrix, dim, "Givens");
}

void CuQuantumVectorPolicyFloat::ApplySWAPalpha(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                calc_type val, index_t dim, bool diff) {
    if (diff) {
        Base::ApplySWAPalpha(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    float angle_pi = static_cast<float>(M_PI) * static_cast<float>(val);
    std::complex<float> e = {cosf(angle_pi), sinf(angle_pi)};
    std::complex<float> a = (1.0f + e) / 2.0f;
    std::complex<float> b = (1.0f - e) / 2.0f;
    // Col Major for {{1,0,0,0}, {0,a,b,0}, {0,b,a,0}, {0,0,0,1}}
    const std::complex<float> swap_alpha_matrix[] = {
        {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, a, b, {0.0f, 0.0f}, {0.0f, 0.0f}, b, a,
        {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}};
    apply_two_qubit_matrix(qs_p, objs, ctrls, swap_alpha_matrix, dim, "SWAPalpha");
}

template struct GPUVectorPolicyBase<CuQuantumVectorPolicyFloat, float>;

}  // namespace mindquantum::sim::vector::detail
