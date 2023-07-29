/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#ifndef MINDQUANTUM_BASE_TYPES_HPP_
#define MINDQUANTUM_BASE_TYPES_HPP_

#include <cmath>

#include <chrono>
#include <complex>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mindquantum {
#define PRECISION     1e-8
#define COS1_2(theta) static_cast<T>(cos((theta) / 2))
#define SIN1_2(theta) static_cast<T>(sin((theta) / 2))
#define ITER(p, obj)                                                                                                   \
    auto(p) = (obj).begin();                                                                                           \
    (p) != (obj).end();                                                                                                \
    (p)++

using Index = std::size_t;
using qbit_t = int64_t;
using qbits_t = std::vector<qbit_t>;
using index_t = std::size_t;

template <typename T>
using VT = std::vector<T>;

template <typename T>
using VVT = VT<VT<T>>;

template <typename T>
using VVVT = VT<VVT<T>>;

template <typename T>
using CT = std::complex<T>;

template <typename T>
using CTP = CT<T> *;

template <typename T>
using MST = std::map<std::string, T>;

using SS = std::set<std::string>;
using VS = VT<std::string>;
using PauliWord = std::pair<Index, char>;

template <typename T>
using PauliTerm = std::pair<VT<PauliWord>, T>;

using TimePoint = std::chrono::steady_clock::time_point;

TimePoint NOW();

int TimeDuration(TimePoint start, TimePoint end);

// void Duration()
struct PauliMask {
    Index mask_x = 0;
    Index mask_y = 0;
    Index mask_z = 0;
    Index num_x = 0;
    Index num_y = 0;
    Index num_z = 0;
};

constexpr const char kNThreads[] = "n_threads";
constexpr const char kNQubits[] = "n_qubits";
constexpr const char kParamNames[] = "param_names";
constexpr const char kEncoderParamsNames[] = "encoder_params_names";
constexpr const char kAnsatzParamsNames[] = "ansatz_params_names";
constexpr const char kGateNames[] = "gate_names";
constexpr const char kGateMatrix[] = "gate_matrix";
constexpr const char kGateObjQubits[] = "gate_obj_qubits";
constexpr const char kGateCtrlQubits[] = "gate_ctrl_qubits";
constexpr const char kGateThetas[] = "gate_thetas";
constexpr const char kNPG[] = "npg";
constexpr const char kGateParamsNames[] = "gate_params_names";
constexpr const char kGateCoeff[] = "gate_coeff";
constexpr const char kGateRequiresGrad[] = "gate_requires_grad";
constexpr const char kHamsPauliCoeff[] = "hams_pauli_coeff";
constexpr const char kHamsPauliWord[] = "hams_pauli_word";
constexpr const char kHamsPauliQubit[] = "hams_pauli_qubit";
constexpr const char kHowTo[] = "how_to";
constexpr const char kHamsSparseData[] = "hams_sparse_data";
constexpr const char kHamsSparseIndice[] = "hams_sparse_indice";
constexpr const char kHamsSparseIndptr[] = "hams_sparse_indptr";
constexpr const char kIsProjector[] = "is_projector";
constexpr const char kProjectors[] = "projectors";

enum SparseHow : int64_t {
    ORIGIN = 0,
    BACKEND,
    FRONTEND,
};
enum HermitianProp : int64_t {
    SELFHERMITIAN = 0,
    DOHERMITIAN,
    PARAMSOPPOSITE,
};
constexpr const char gX[] = "X";
constexpr const char gY[] = "Y";
constexpr const char gZ[] = "Z";
constexpr const char gI[] = "I";
constexpr const char gH[] = "H";
constexpr const char gT[] = "T";
constexpr const char gS[] = "S";
constexpr const char gCNOT[] = "CNOT";
constexpr const char gCZ[] = "CZ";
constexpr const char gSWAP[] = "SWAP";
constexpr const char gISWAP[] = "ISWAP";
constexpr const char gRX[] = "RX";
constexpr const char gRY[] = "RY";
constexpr const char gRZ[] = "RZ";
constexpr const char gGP[] = "GP";
constexpr const char gPS[] = "PS";
constexpr const char gRXX[] = "Rxx";
constexpr const char gRYY[] = "Ryy";
constexpr const char gRZZ[] = "Rzz";
constexpr const char gU3[] = "U3";
constexpr const char gFSim[] = "FSim";
constexpr const char cPL[] = "PL";   // Pauli channel
constexpr const char cAD[] = "ADC";  // amplitude damping channel
constexpr const char cPD[] = "PDC";  // phase damping channel

/**
 * The qubits number threshold for a simulator to use OpenMP or not.
 * For a small system, it is move efficient to run simulation without OpenMP.
 * Qulacs also use this strategy:
 * https://github.com/qulacs/qulacs/blob/8cd29d4c1d7836c37b32b42a2516d1fbcd41535a/src/csim/update_ops_pauli_multi.c#L40
 *
 */
static constexpr Index nQubitTh = 13;
}  // namespace mindquantum
#endif  // MINDQUANTUM_BASE_TYPES_HPP_
