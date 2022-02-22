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

#ifndef MINDQUANTUM_UTILS_H_
#define MINDQUANTUM_UTILS_H_

#include <stdint.h>
#ifdef ENABLE_OPENMP
#    include <omp.h>
#endif  // ENABLE_OPENMP  // NOLINT

#ifdef _MSC_VER
#    include <intrin.h>
#else
#    include <x86intrin.h>
#endif  // _MSC_VER

#include <complex>
#include <cstdlib>
#include <ctime>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/type.h"

namespace mindquantum {
extern const VT<CT<MT>> POLAR;
template <typename T, typename ST>
CT<T> ComplexInnerProduct(const ST *v1, const ST *v2, Index len) {
    // len is (1UL>>n_qubits)*2
    ST real_part = 0;
    ST imag_part = 0;
    auto size = len / 2;
#pragma omp parallel for reduction(+ : real_part, imag_part)
    for (Index i = 0; i < size; i++) {
        real_part += v1[2 * i] * v2[2 * i] + v1[2 * i + 1] * v2[2 * i + 1];
        imag_part += v1[2 * i] * v2[2 * i + 1] - v1[2 * i + 1] * v2[2 * i];
    }

    CT<T> result = {static_cast<T>(real_part), static_cast<T>(imag_part)};
    return result;
}

template <typename T, typename ST>
CT<T> ComplexInnerProductWithControl(const ST *v1, const ST *v2, Index len, Index ctrlmask) {
    // len is (1UL>>n_qubits)*2
    ST real_part = 0;
    ST imag_part = 0;
    auto size = len / 2;
#pragma omp parallel for reduction(+ : real_part, imag_part)
    for (Index i = 0; i < size; i++) {
        if ((i & ctrlmask) == ctrlmask) {
            real_part += v1[2 * i] * v2[2 * i] + v1[2 * i + 1] * v2[2 * i + 1];
            imag_part += v1[2 * i] * v2[2 * i + 1] - v1[2 * i + 1] * v2[2 * i];
        }
    }
    CT<T> result = {static_cast<T>(real_part), static_cast<T>(imag_part)};
    return result;
}

Index GetControlMask(const VT<Index> &ctrls);

PauliMask GetPauliMask(const VT<PauliWord> &pws);

#ifdef _MSC_VER
inline uint32_t CountOne(uint32_t n) {
    return __popcnt(n);
}
inline uint64_t CountOne(uint64_t n) {
    return __popcnt64(n);
}
inline uint32_t CountOne(int32_t n) {
    return CountOne(uint32_t(n));
}
inline uint64_t CountOne(int64_t n) {
    return CountOne(uint64_t(n));
}
#else
inline uint32_t CountOne(uint32_t n) {
    int result;
    asm("popcnt %1,%0" : "=r"(result) : "r"(n));
    return result;
}

inline uint64_t CountOne(int64_t n) {
    uint32_t *p = reinterpret_cast<uint32_t *>(&n);
    return CountOne(p[0]) + CountOne(p[1]);
}
#endif  // _MSC_VER

// inline int CountOne(uint64_t n) {
//   uint8_t *p = reinterpret_cast<uint8_t *>(&n);
//   return POPCNTTABLE[p[0]] + POPCNTTABLE[p[1]] + POPCNTTABLE[p[2]] +
//          POPCNTTABLE[p[3]] + POPCNTTABLE[p[4]] + POPCNTTABLE[p[5]] +
//          POPCNTTABLE[p[6]] + POPCNTTABLE[p[7]];
// }

// inline int CountOne(uint32_t n) {
//   uint8_t *p = reinterpret_cast<uint8_t *>(&n);
//   return POPCNTTABLE[p[0]] + POPCNTTABLE[p[1]] + POPCNTTABLE[p[2]] +
//          POPCNTTABLE[p[3]];
// }

template <typename T>
PauliTerm<T> GenerateRandomPauliTerm(Index n_qubits) {
    std::default_random_engine e(std::clock());
    std::uniform_real_distribution<T> ut(-1.0, 1.0);
    auto coeff = ut(e);
    std::uniform_int_distribution<char> uit(0, 3);
    VT<PauliWord> pws;
    for (Index i = 0; i < n_qubits; i++) {
        auto p = uit(e);
        if (p != 3) {
            pws.push_back(std::make_pair(i, (p + 'X')));
        }
    }
    return std::make_pair(pws, coeff);
}

template <typename T>
void ShowPauliTerm(const PauliTerm<T> &pt) {
    std::cout << pt.second << " [";
    for (Index i = 0; i < static_cast<Index>(pt.first.size()); i++) {
        auto &pw = pt.first[i];
        std::cout << pw.second << pw.first;
        if (i != static_cast<Index>(pt.first.size()) - 1) {
            std::cout << " ";
        }
    }
    std::cout << "]" << std::endl;
}

TimePoint NOW();
int TimeDuration(TimePoint start, TimePoint end);

template <typename T>
void PrintVec(T *vec, size_t len) {
    auto cvec = reinterpret_cast<CTP<T>>(vec);
    for (size_t i = 0; i < len / 2; i++) {
        std::cout << cvec[i] << std::endl;
    }
}
}  // namespace mindquantum
#endif  // MINDQUANTUM_UTILS_H_
