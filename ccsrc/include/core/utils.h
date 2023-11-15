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

#ifndef MINDQUANTUM_UTILS_HPP_
#define MINDQUANTUM_UTILS_HPP_

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "config/config.h"
#include "config/openmp.h"
#include "config/popcnt.h"
#include "core/mq_base_types.h"

namespace mindquantum {
#ifndef MQ_DO_PRAGMA
#    define MQ_DO_PRAGMA(x) _Pragma(#    x)
#endif  // !MQ_DO_PRAGMA

#define THRESHOLD_OMP(omp_pragma, n, n_th, ...)                                                                        \
    if ((n) < (n_th)) {                                                                                                \
        __VA_ARGS__                                                                                                    \
    } else {                                                                                                           \
        omp_pragma __VA_ARGS__                                                                                         \
    }
#define THRESHOLD_OMP_FOR(n, n_th, ...)                                                                                \
    THRESHOLD_OMP(MQ_DO_PRAGMA(omp parallel for schedule(static)), n, n_th, __VA_ARGS__)

extern const VT<CT<double>> POLAR;
template <typename T, typename ST>
CT<T> ComplexInnerProduct(const ST *v1, const ST *v2, Index len) {
    // len is (static_cast<uint64_t>(1)>>n_qubits)*2
    ST real_part = 0;
    ST imag_part = 0;
    auto size = len / 2;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+ : real_part, imag_part)), len, static_cast<uint64_t>(2) << nQubitTh,
                     for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(size); i++) {
                         real_part += v1[2 * i] * v2[2 * i] + v1[2 * i + 1] * v2[2 * i + 1];
                         imag_part += v1[2 * i] * v2[2 * i + 1] - v1[2 * i + 1] * v2[2 * i];
                     })
    CT<T> result = {static_cast<T>(real_part), static_cast<T>(imag_part)};
    return result;
}

template <typename T, typename ST>
CT<T> ComplexInnerProductWithControl(const ST *v1, const ST *v2, Index len, Index ctrl_mask) {
    // len is (static_cast<uint64_t>(1)>>n_qubits)*2
    ST real_part = 0;
    ST imag_part = 0;
    auto size = len / 2;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+ : real_part, imag_part)), len, static_cast<uint64_t>(2) << nQubitTh,
                     for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(size); i++) {
                         if ((i & ctrl_mask) == ctrl_mask) {
                             real_part += v1[2 * i] * v2[2 * i] + v1[2 * i + 1] * v2[2 * i + 1];
                             imag_part += v1[2 * i] * v2[2 * i + 1] - v1[2 * i + 1] * v2[2 * i];
                         }
                     })
    CT<T> result = {static_cast<T>(real_part), static_cast<T>(imag_part)};
    return result;
}

Index GetControlMask(const qbits_t &ctrls);

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
inline uint32_t CountLeadingZero(uint32_t n) {
    return __lzcnt(n);
}
inline uint64_t CountLeadingZero(uint64_t n) {
    return __lzcnt64(n);
}
inline uint32_t CountLeadingZero(int32_t n) {
    return __lzcnt(uint32_t(n));
}
inline uint64_t CountLeadingZero(int64_t n) {
    return __lzcnt64(uint64_t(n));
}
#else

inline uint32_t CountOne(uint32_t n) {
    return __builtin_popcount(n);
}

inline uint64_t CountOne(uint64_t n) {
    return __builtin_popcount(n) + __builtin_popcount(n >> 32);
}
inline uint32_t CountLeadingZero(uint32_t n) {
    return __builtin_clzll(n);
}
inline uint64_t CountLeadingZero(uint64_t n) {
    return __builtin_clzll(n);
}
inline uint32_t CountLeadingZero(int32_t n) {
    return __builtin_clzll(uint32_t(n));
}
inline uint64_t CountLeadingZero(int64_t n) {
    return __builtin_clzll(uint64_t(n));
}
#endif  // _MSC_VER

template <typename T>
PauliTerm<T> GenerateRandomPauliTerm(Index n_qubits) {
    std::default_random_engine e(std::clock());
    std::uniform_real_distribution<T> ut(-1.0, 1.0);
    auto coeff = ut(e);
    std::uniform_int_distribution<int16_t> uit(0, 3);
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
    auto c_vec = reinterpret_cast<CTP<T>>(vec);
    for (size_t i = 0; i < len / 2; i++) {
        std::cout << c_vec[i] << std::endl;
    }
}

void safe_copy(void *dest, size_t dest_size, const void *src, size_t count);
}  // namespace mindquantum
#endif  // MINDQUANTUM_UTILS_HPP_
