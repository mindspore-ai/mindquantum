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

#ifndef MINDQUANTUM_HAMILTONIAN_HAMILTONIAN_H_
#define MINDQUANTUM_HAMILTONIAN_HAMILTONIAN_H_
#include <memory>

#include "core/sparse/algo.h"
#include "core/utils.h"

namespace mindquantum {
using mindquantum::sparse::CsrHdMatrix;
using mindquantum::sparse::SparseHamiltonian;
using mindquantum::sparse::TransposeCsrHdMatrix;

template <typename T>
struct Hamiltonian {
    int64_t how_to_ = 0;
    Index n_qubits_ = 0;
    VT<PauliTerm<T>> ham_;
    std::shared_ptr<CsrHdMatrix<T>> ham_sparse_main_;
    std::shared_ptr<CsrHdMatrix<T>> ham_sparse_second_;

    Hamiltonian() = default;

    explicit Hamiltonian(const VT<PauliTerm<T>> &ham) : how_to_(ORIGIN), ham_(ham) {
    }

    Hamiltonian(const VT<PauliTerm<T>> &ham, Index n_qubits) : how_to_(BACKEND), n_qubits_(n_qubits), ham_(ham) {
        if (n_qubits_ > 16) {
            std::cout << "Sparsing hamiltonian ..." << std::endl;
        }
        ham_sparse_main_ = SparseHamiltonian(ham_, n_qubits_);
        ham_sparse_second_ = TransposeCsrHdMatrix(ham_sparse_main_);
        if (n_qubits_ > 16) {
            std::cout << "Sparsing hamiltonian finished!" << std::endl;
        }
    }

    Hamiltonian(std::shared_ptr<CsrHdMatrix<T>> csr_mat, Index n_qubits)
        : n_qubits_(n_qubits), how_to_(FRONTEND), ham_sparse_main_(csr_mat) {
    }
};
}  // namespace mindquantum
#endif  // MINDQUANTUM_HAMILTONIAN_HAMILTONIAN_H_
