/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDQUANTUM_TYPE_H_
#define MINDQUANTUM_TYPE_H_
// #ifndef MQONLY
// #include "backend/kernel_compiler/cpu/quantum/quantum_simulator/popcnt.h"
// #else
// #include "./popcnt.h"
// #endif

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
#define COS1_2(theta) static_cast<T>(cos(theta / 2))
#define SIN1_2(theta) static_cast<T>(sin(theta / 2))

using Index = int64_t;
#ifdef FLOAT
using MT = float;
#else
using MT = double;
#endif

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

const char kNThreads[] = "n_threads";
const char kNQubits[] = "n_qubits";
const char kParamNames[] = "param_names";
const char kEncoderParamsNames[] = "encoder_params_names";
const char kAnsatzParamsNames[] = "ansatz_params_names";
const char kGateNames[] = "gate_names";
const char kGateMatrix[] = "gate_matrix";
const char kGateObjQubits[] = "gate_obj_qubits";
const char kGateCtrlQubits[] = "gate_ctrl_qubits";
const char kGateThetas[] = "gate_thetas";
const char kNPG[] = "npg";
const char kGateParamsNames[] = "gate_params_names";
const char kGateCoeff[] = "gate_coeff";
const char kGateRequiresGrad[] = "gate_requires_grad";
const char kHamsPauliCoeff[] = "hams_pauli_coeff";
const char kHamsPauliWord[] = "hams_pauli_word";
const char kHamsPauliQubit[] = "hams_pauli_qubit";
const char kHowTo[] = "how_to";
const char kHamsSparseData[] = "hams_sparse_data";
const char kHamsSparseIndice[] = "hams_sparse_indice";
const char kHamsSparseIndptr[] = "hams_sparse_indptr";
const char kIsProjector[] = "is_projector";
const char kProjectors[] = "projectors";

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
const char gX[] = "X";
const char gY[] = "Y";
const char gZ[] = "Z";
const char gI[] = "I";
const char gH[] = "H";
const char gT[] = "T";
const char gS[] = "S";
const char gCNOT[] = "CNOT";
const char gCZ[] = "CZ";
const char gSWAP[] = "SWAP";
const char gISWAP[] = "ISWAP";
const char gRX[] = "RX";
const char gRY[] = "RY";
const char gRZ[] = "RZ";
const char gGP[] = "GP";
const char gPS[] = "PS";
const char gXX[] = "XX";
const char gYY[] = "YY";
const char gZZ[] = "ZZ";
const char cPL[] = "PL";    // Pauli channel
const char cAD[] = "AD";    // amplitude damping channel
const char cPD[] = "PD";    // phase damping channel
}  // namespace mindquantum
#endif  // MINDQUANTUM_TYPE_H_
