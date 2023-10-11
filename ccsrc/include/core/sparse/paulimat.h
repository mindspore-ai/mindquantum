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
#ifndef MINDQUANTUM_SPARSE_PAULI_MAT_H_
#define MINDQUANTUM_SPARSE_PAULI_MAT_H_

#include "core/utils.h"

namespace mindquantum {
namespace sparse {
template <typename T>
struct PauliMat {
    char *coeff_;
    Index *col_;
    Index n_qubits_;
    Index dim_;
    T p_;

    inline void FreeMemory() {
        if (coeff_ != nullptr) {
            free(coeff_);
        }
        if (col_ != nullptr) {
            free(col_);
        }
    }
    void Reset() {
        FreeMemory();
        coeff_ = nullptr;
        col_ = nullptr;
    }
    ~PauliMat() {
        FreeMemory();
    }
    PauliMat() : coeff_(nullptr), col_(nullptr), n_qubits_(0), dim_(0) {
    }
    PauliMat(const PauliTerm<T> pt, Index n_qubits) : n_qubits_(n_qubits), p_(pt.second) {
        dim_ = (static_cast<uint64_t>(1) << n_qubits_);
        coeff_ = reinterpret_cast<char *>(malloc(sizeof(char) * dim_));
        col_ = reinterpret_cast<Index *>(malloc(sizeof(Index) * dim_));
        auto mask = GetPauliMask(pt.first);
        auto mask_f = mask.mask_x | mask.mask_y;
        THRESHOLD_OMP_FOR(
            dim_, static_cast<uint64_t>(1) << nQubitTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim_); i++) {
                auto j = (i ^ mask_f);
                col_[i] = j;
                auto axis2power = CountOne(static_cast<uint64_t>(i & mask.mask_z));  // -1
                auto axis3power = CountOne(static_cast<uint64_t>(i & mask.mask_y));  // -1j
                // (-1)^a2*(-1j)^a3*(1j)^a1=(1j)^2a2*(1j)^3a3*(1j)^a1=(1j)^(a1+2*a2+3*a3)
                coeff_[j] = static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3);
            })
    }
    void PrintInfo() {
        std::cout << "<--Pauli Matrix with Dimension: ";
        std::cout << dim_ << " X " << dim_ << std::endl;
        std::cout << "   Data:\n   ";
        for (Index i = 0; i < dim_; i++) {
            std::cout << POLAR[coeff_[i]];
            if (i != dim_ - 1) {
                std::cout << ",";
            }
        }

        std::cout << "\n   Col:\n   ";
        for (Index i = 0; i < dim_; i++) {
            std::cout << col_[i];
            if (i != dim_ - 1) {
                std::cout << ",";
            }
        }
        std::cout << "-->\n\n";
    }
};
}  // namespace sparse
}  // namespace mindquantum
#endif  // MINDQUANTUM_SPARSE_PAULI_MAT_H_
