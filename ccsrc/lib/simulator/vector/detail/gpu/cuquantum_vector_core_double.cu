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
#include "simulator/vector/detail/cuquantum_vector_double_policy.cuh"

namespace mindquantum::sim::vector::detail {

namespace {
const std::complex<double> h_matrix_d[] = {{M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0}, {-M_SQRT1_2, 0.0}};
const std::complex<double> x_matrix_d[] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
const std::complex<double> y_matrix_d[] = {{0.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {0.0, 0.0}};
const std::complex<double> z_matrix_d[] = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}};
const std::complex<double> s_matrix_d[] = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}};
const std::complex<double> sdag_matrix_d[] = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, -1.0}};
const double t_angle_factor_d = M_SQRT1_2;
const std::complex<double> t_matrix_d[] = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {t_angle_factor_d, t_angle_factor_d}};
const std::complex<double> tdag_matrix_d[] = {
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {t_angle_factor_d, -t_angle_factor_d}};
const std::complex<double> sx_matrix_d[] = {{0.5, 0.5}, {0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}};
const std::complex<double> sxdag_matrix_d[] = {{0.5, -0.5}, {0.5, 0.5}, {0.5, 0.5}, {0.5, -0.5}};
const std::complex<double> swap_matrix_d[] = {  // Col Major for {{1,0,0,0},{0,0,1,0},{0,1,0,0},{0,0,0,1}}
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
}  // namespace

custatevecHandle_t CuQuantumVectorPolicyDouble::handle_ = nullptr;

inline void checkCuStateVecError_double(custatevecStatus_t status, const std::string& func_name) {
    if (status != CUSTATEVEC_STATUS_SUCCESS) {
        throw std::runtime_error("cuStateVec Error in function " + func_name + ": "
                                 + CUSTATEVEC_API.custatevecGetErrorString(status));
    }
}

void CuQuantumVectorPolicyDouble::Init() {
    if (!CuQuantumLoader::GetInstance().IsAvailable()) {
        throw std::runtime_error("cuQuantum backend is not available. Please ensure cuQuantum is installed correctly.");
    }
    if (handle_ == nullptr) {
        checkCuStateVecError_double(CUSTATEVEC_API.custatevecCreate(&handle_), "custatevecCreate");
    }
}

void CuQuantumVectorPolicyDouble::Destroy() {
    if (handle_ != nullptr) {
        checkCuStateVecError_double(CUSTATEVEC_API.custatevecDestroy(handle_), "custatevecDestroy");
        handle_ = nullptr;
    }
}

void CuQuantumVectorPolicyDouble::ApplyH(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }

    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));
    auto sv_data_type = CUDA_C_64F;
    void* sv_ptr = reinterpret_cast<void*>(qs);
    auto matrix_data_type = CUDA_C_64F;
    auto layout = CUSTATEVEC_MATRIX_LAYOUT_COL;
    int adjoint = 0;

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError_double(
        CUSTATEVEC_API.custatevecApplyMatrix(handle_, sv_ptr, sv_data_type, n_qubits, h_matrix_d, matrix_data_type,
                                             layout, adjoint, target_bits.data(), target_bits.size(), ctrl_ptr, nullptr,
                                             control_bits.size(), CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
        "custatevecApplyMatrix (H)");
}

void CuQuantumVectorPolicyDouble::ApplyX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    Init();
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = Base::InitState(dim);
    }

    uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));
    auto sv_data_type = CUDA_C_64F;
    void* sv_ptr = reinterpret_cast<void*>(qs);
    auto matrix_data_type = CUDA_C_64F;
    auto layout = CUSTATEVEC_MATRIX_LAYOUT_COL;
    int adjoint = 0;

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError_double(
        CUSTATEVEC_API.custatevecApplyMatrix(handle_, sv_ptr, sv_data_type, n_qubits, x_matrix_d, matrix_data_type,
                                             layout, adjoint, target_bits.data(), target_bits.size(), ctrl_ptr, nullptr,
                                             control_bits.size(), CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
        "custatevecApplyMatrix (X)");
}

void CuQuantumVectorPolicyDouble::ApplyRX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
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
    auto sv_data_type = CUDA_C_64F;
    void* sv_ptr = reinterpret_cast<void*>(qs);

    // Angle transformation: RX(θ) = exp(-i θ/2 X), angle = -θ/2.
    double angle_for_pauli_rot = -static_cast<double>(val) / 2.0;

    std::vector<int32_t> target_bits = {static_cast<int32_t>(objs[0])};
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_X};
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError_double(CUSTATEVEC_API.custatevecApplyPauliRotation(
                                    handle_, sv_ptr, sv_data_type, n_qubits, angle_for_pauli_rot, paulis,
                                    target_bits.data(), target_bits.size(), ctrl_ptr, nullptr, control_bits.size()),
                                "custatevecApplyPauliRotation (RX)");
}

void CuQuantumVectorPolicyDouble::ApplyRzz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
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
    auto sv_data_type = CUDA_C_64F;
    void* sv_ptr = reinterpret_cast<void*>(qs);

    // Angle transformation: Rzz(θ) = exp(-i θ/2 ZZ), but custatevecApplyPauliRotation implements exp(i angle P).
    // So, we need angle = -θ/2.
    double angle_for_pauli_rot = -static_cast<double>(val) / 2.0;

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_Z, CUSTATEVEC_PAULI_Z};

    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError_double(
        CUSTATEVEC_API.custatevecApplyPauliRotation(handle_, sv_ptr, sv_data_type, n_qubits,
                                                    angle_for_pauli_rot,  // Use transformed angle
                                                    paulis, target_bits.data(), target_bits.size(), ctrl_ptr, nullptr,
                                                    control_bits.size()),
        "custatevecApplyPauliRotation (Rzz)");
}

#define APPLY_STD_SINGLE_QUBIT_MATRIX_DOUBLE(GATE_NAME, MATRIX)                                                        \
    void CuQuantumVectorPolicyDouble::Apply##GATE_NAME(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,   \
                                                       index_t dim) {                                                  \
        Init();                                                                                                        \
        auto& qs = *qs_p;                                                                                              \
        if (qs == nullptr) {                                                                                           \
            qs = Base::InitState(dim);                                                                                 \
        }                                                                                                              \
        uint32_t n_qubits = static_cast<uint32_t>(std::log2(static_cast<double>(dim)));                                \
        auto sv_data_type = CUDA_C_64F;                                                                                \
        void* sv_ptr = reinterpret_cast<void*>(qs);                                                                    \
        auto matrix_data_type = CUDA_C_64F;                                                                            \
        auto layout = CUSTATEVEC_MATRIX_LAYOUT_COL;                                                                    \
        int adjoint = 0;                                                                                               \
        std::vector<int32_t> target_bits(objs.begin(), objs.end());                                                    \
        std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());                                                 \
        const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();                                \
        checkCuStateVecError_double(CUSTATEVEC_API.custatevecApplyMatrix(                                              \
                                        handle_, sv_ptr, sv_data_type, n_qubits, MATRIX, matrix_data_type, layout,     \
                                        adjoint, target_bits.data(), target_bits.size(), ctrl_ptr, nullptr,            \
                                        control_bits.size(), CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),                  \
                                    "custatevecApplyMatrix (" #GATE_NAME ")");                                         \
    }

APPLY_STD_SINGLE_QUBIT_MATRIX_DOUBLE(Y, y_matrix_d)
APPLY_STD_SINGLE_QUBIT_MATRIX_DOUBLE(Z, z_matrix_d)
APPLY_STD_SINGLE_QUBIT_MATRIX_DOUBLE(SGate, s_matrix_d)
APPLY_STD_SINGLE_QUBIT_MATRIX_DOUBLE(Sdag, sdag_matrix_d)
APPLY_STD_SINGLE_QUBIT_MATRIX_DOUBLE(T, t_matrix_d)
APPLY_STD_SINGLE_QUBIT_MATRIX_DOUBLE(Tdag, tdag_matrix_d)
APPLY_STD_SINGLE_QUBIT_MATRIX_DOUBLE(SX, sx_matrix_d)
APPLY_STD_SINGLE_QUBIT_MATRIX_DOUBLE(SXdag, sxdag_matrix_d)

void CuQuantumVectorPolicyDouble::ApplyRY(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
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
    auto sv_data_type = CUDA_C_64F;
    void* sv_ptr = reinterpret_cast<void*>(qs);

    // Angle transformation: RY(θ) = exp(-i θ/2 Y), angle = -θ/2.
    double angle_for_pauli_rot = -static_cast<double>(val) / 2.0;

    std::vector<int32_t> target_bits = {static_cast<int32_t>(objs[0])};
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_Y};
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError_double(CUSTATEVEC_API.custatevecApplyPauliRotation(
                                    handle_, sv_ptr, sv_data_type, n_qubits, angle_for_pauli_rot, paulis,
                                    target_bits.data(), target_bits.size(), ctrl_ptr, nullptr, control_bits.size()),
                                "custatevecApplyPauliRotation (RY)");
}

void CuQuantumVectorPolicyDouble::ApplyRZ(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
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
    auto sv_data_type = CUDA_C_64F;
    void* sv_ptr = reinterpret_cast<void*>(qs);

    // Angle transformation: RZ(θ) = exp(-i θ/2 Z), angle = -θ/2.
    double angle_for_pauli_rot = -static_cast<double>(val) / 2.0;

    std::vector<int32_t> target_bits = {static_cast<int32_t>(objs[0])};
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_Z};
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError_double(CUSTATEVEC_API.custatevecApplyPauliRotation(
                                    handle_, sv_ptr, sv_data_type, n_qubits, angle_for_pauli_rot, paulis,
                                    target_bits.data(), target_bits.size(), ctrl_ptr, nullptr, control_bits.size()),
                                "custatevecApplyPauliRotation (RZ)");
}

void CuQuantumVectorPolicyDouble::ApplyPS(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
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
    auto sv_data_type = CUDA_C_64F;
    void* sv_ptr = reinterpret_cast<void*>(qs);
    auto matrix_data_type = CUDA_C_64F;
    auto layout = CUSTATEVEC_MATRIX_LAYOUT_COL;
    int adjoint = 0;

    double angle = static_cast<double>(val);
    double cos_val = cos(angle);
    double sin_val = sin(angle);
    const std::complex<double> ps_matrix_d[] = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {cos_val, sin_val}};

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError_double(
        CUSTATEVEC_API.custatevecApplyMatrix(handle_, sv_ptr, sv_data_type, n_qubits, ps_matrix_d, matrix_data_type,
                                             layout, adjoint, target_bits.data(), target_bits.size(), ctrl_ptr, nullptr,
                                             control_bits.size(), CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
        "custatevecApplyMatrix (PS)");
}

void CuQuantumVectorPolicyDouble::ApplyGP(qs_data_p_t* qs_p, qbit_t obj_qubit, const qbits_t& ctrls, calc_type val,
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
    auto sv_data_type = CUDA_C_64F;
    void* sv_ptr = reinterpret_cast<void*>(qs);
    auto matrix_data_type = CUDA_C_64F;
    auto layout = CUSTATEVEC_MATRIX_LAYOUT_COL;
    int adjoint = 0;

    double angle = static_cast<double>(val);
    double cos_val = cos(angle);
    double sin_val = sin(angle);
    const std::complex<double> gp_matrix_d[] = {{cos_val, -sin_val}, {0.0, 0.0}, {0.0, 0.0}, {cos_val, -sin_val}};

    std::vector<int32_t> target_bits = {static_cast<int32_t>(obj_qubit)};
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError_double(
        CUSTATEVEC_API.custatevecApplyMatrix(handle_, sv_ptr, sv_data_type, n_qubits, gp_matrix_d, matrix_data_type,
                                             layout, adjoint, target_bits.data(), target_bits.size(), ctrl_ptr, nullptr,
                                             control_bits.size(), CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
        "custatevecApplyMatrix (GP)");
}

void CuQuantumVectorPolicyDouble::ApplyRxx(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
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
    auto sv_data_type = CUDA_C_64F;
    void* sv_ptr = reinterpret_cast<void*>(qs);

    double angle_for_pauli_rot = -static_cast<double>(val) / 2.0;

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_X, CUSTATEVEC_PAULI_X};
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError_double(CUSTATEVEC_API.custatevecApplyPauliRotation(
                                    handle_, sv_ptr, sv_data_type, n_qubits, angle_for_pauli_rot, paulis,
                                    target_bits.data(), target_bits.size(), ctrl_ptr, nullptr, control_bits.size()),
                                "custatevecApplyPauliRotation (Rxx)");
}

void CuQuantumVectorPolicyDouble::ApplyRyy(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
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
    auto sv_data_type = CUDA_C_64F;
    void* sv_ptr = reinterpret_cast<void*>(qs);

    double angle_for_pauli_rot = -static_cast<double>(val) / 2.0;

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_Y, CUSTATEVEC_PAULI_Y};
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError_double(CUSTATEVEC_API.custatevecApplyPauliRotation(
                                    handle_, sv_ptr, sv_data_type, n_qubits, angle_for_pauli_rot, paulis,
                                    target_bits.data(), target_bits.size(), ctrl_ptr, nullptr, control_bits.size()),
                                "custatevecApplyPauliRotation (Ryy)");
}

void CuQuantumVectorPolicyDouble::apply_two_qubit_matrix(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                         const std::complex<double>* matrix, index_t dim,
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
    auto sv_data_type = CUDA_C_64F;
    void* sv_ptr = reinterpret_cast<void*>(qs);
    auto matrix_data_type = CUDA_C_64F;
    auto layout = CUSTATEVEC_MATRIX_LAYOUT_COL;
    int adjoint = 0;

    std::vector<int32_t> target_bits(objs.begin(), objs.end());
    std::vector<int32_t> control_bits(ctrls.begin(), ctrls.end());
    const int32_t* ctrl_ptr = control_bits.empty() ? nullptr : control_bits.data();

    checkCuStateVecError_double(
        CUSTATEVEC_API.custatevecApplyMatrix(handle_, sv_ptr, sv_data_type, n_qubits, matrix, matrix_data_type, layout,
                                             adjoint, target_bits.data(), target_bits.size(), ctrl_ptr, nullptr,
                                             control_bits.size(), CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
        "custatevecApplyMatrix (" + gate_name + ")");
}

void CuQuantumVectorPolicyDouble::ApplySWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    apply_two_qubit_matrix(qs_p, objs, ctrls, swap_matrix_d, dim, "SWAP");
}

void CuQuantumVectorPolicyDouble::ApplyISWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                             bool daggered, index_t dim) {
    double im_val = daggered ? -1.0 : 1.0;
    const std::complex<double> iswap_matrix_d[] = {
        {1.0, 0.0}, {0.0, 0.0},    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, im_val}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, im_val}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},    {1.0, 0.0}};
    apply_two_qubit_matrix(qs_p, objs, ctrls, iswap_matrix_d, dim, daggered ? "ISWAPdg" : "ISWAP");
}

void CuQuantumVectorPolicyDouble::ApplyRxy(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                           index_t dim, bool diff) {
    if (diff) {
        Base::ApplyRxy(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    double angle = static_cast<double>(val);
    double c = cos(angle / 2.0);
    double s = sin(angle / 2.0);
    const std::complex<double> rxy_matrix_d[] = {{c, 0.0},  {0.0, 0.0}, {0.0, 0.0}, {s, 0.0},  {0.0, 0.0}, {c, 0.0},
                                                 {s, 0.0},  {0.0, 0.0}, {0.0, 0.0}, {-s, 0.0}, {c, 0.0},   {0.0, 0.0},
                                                 {-s, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {c, 0.0}};
    apply_two_qubit_matrix(qs_p, objs, ctrls, rxy_matrix_d, dim, "Rxy");
}

void CuQuantumVectorPolicyDouble::ApplyRxz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                           index_t dim, bool diff) {
    if (diff) {
        Base::ApplyRxz(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    double angle = static_cast<double>(val);
    double c = cos(angle / 2.0);
    double s = sin(angle / 2.0);
    const std::complex<double> rxz_matrix_d[] = {{c, 0.0},   {0.0, -s},  {0.0, 0.0}, {0.0, 0.0}, {0.0, -s}, {c, 0.0},
                                                 {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {c, 0.0},  {0.0, s},
                                                 {0.0, 0.0}, {0.0, 0.0}, {0.0, s},   {c, 0.0}};
    apply_two_qubit_matrix(qs_p, objs, ctrls, rxz_matrix_d, dim, "Rxz");
}

void CuQuantumVectorPolicyDouble::ApplyRyz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                           index_t dim, bool diff) {
    if (diff) {
        Base::ApplyRyz(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    double angle = static_cast<double>(val);
    double c = cos(angle / 2.0);
    double s = sin(angle / 2.0);
    const std::complex<double> ryz_matrix_d[] = {{c, 0.0},   {s, 0.0},   {0.0, 0.0}, {0.0, 0.0}, {-s, 0.0}, {c, 0.0},
                                                 {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {c, 0.0},  {-s, 0.0},
                                                 {0.0, 0.0}, {0.0, 0.0}, {s, 0.0},   {c, 0.0}};
    apply_two_qubit_matrix(qs_p, objs, ctrls, ryz_matrix_d, dim, "Ryz");
}

void CuQuantumVectorPolicyDouble::ApplyGivens(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                              calc_type val, index_t dim, bool diff) {
    if (diff) {
        Base::ApplyGivens(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    double angle = static_cast<double>(val);
    double c = cos(angle);
    double s = sin(angle);
    const std::complex<double> givens_matrix_d[] = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {c, 0.0},   {s, 0.0},   {0.0, 0.0},
        {0.0, 0.0}, {-s, 0.0},  {c, 0.0},   {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    apply_two_qubit_matrix(qs_p, objs, ctrls, givens_matrix_d, dim, "Givens");
}

void CuQuantumVectorPolicyDouble::ApplySWAPalpha(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                 calc_type val, index_t dim, bool diff) {
    if (diff) {
        Base::ApplySWAPalpha(qs_p, objs, ctrls, val, dim, diff);
        return;
    }
    double angle_pi = M_PI * static_cast<double>(val);
    std::complex<double> e = {cos(angle_pi), sin(angle_pi)};
    std::complex<double> a = (1.0 + e) / 2.0;
    std::complex<double> b = (1.0 - e) / 2.0;
    const std::complex<double> swap_alpha_matrix_d[] = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, a,          b,          {0.0, 0.0},
        {0.0, 0.0}, b,          a,          {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    apply_two_qubit_matrix(qs_p, objs, ctrls, swap_alpha_matrix_d, dim, "SWAPalpha");
}

template struct GPUVectorPolicyBase<CuQuantumVectorPolicyDouble, double>;

}  // namespace mindquantum::sim::vector::detail
