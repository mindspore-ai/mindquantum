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
#ifndef MINDQUANTUM_SPARSE_ALGO_H_
#define MINDQUANTUM_SPARSE_ALGO_H_

#include <memory>

#include "config/openmp.h"
#include "config/type_promotion.h"
#include "core/sparse/csrhdmatrix.h"
#include "core/sparse/paulimat.h"
#include "core/sparse/sparse_utils.h"
#include "core/utils.h"

namespace mindquantum::sparse {
template <typename T>
std::shared_ptr<CsrHdMatrix<T>> TransposeCsrHdMatrix(std::shared_ptr<CsrHdMatrix<T>> a) {
    auto &dim = a->dim_;
    auto &nnz = a->nnz_;
    auto &a_indices = a->indices_;
    auto &a_indptr = a->indptr_;
    auto &a_data = a->data_;
    auto *indices = reinterpret_cast<Index *>(malloc(sizeof(Index) * nnz));
    auto *indptr = reinterpret_cast<Index *>(malloc(sizeof(Index) * (dim + 1)));
    auto data = reinterpret_cast<CTP<T>>(malloc(sizeof(CT<T>) * nnz));
    std::fill(indptr, indptr + dim, 0);
    for (Index n = 0; n < nnz; n++) {
        indptr[a_indices[n]]++;
    }
    for (Index col = 0, sum = 0; col < dim; col++) {
        Index t = indptr[col];
        indptr[col] = sum;
        sum += t;
    }
    indptr[dim] = nnz;
    for (Index row = 0; row < dim; row++) {
        for (Index jj = a_indptr[row]; jj < a_indptr[row + 1]; jj++) {
            Index col = a_indices[jj];
            Index dest = indptr[col];
            indices[dest] = row;
            data[dest] = std::conj(a_data[jj]);
            indptr[col]++;
        }
    }
    for (Index col = 0, last = 0; col <= dim; col++) {
        Index t = indptr[col];
        indptr[col] = last;
        last = t;
    }
    auto c = std::make_shared<CsrHdMatrix<T>>(dim, nnz, indptr, indices, data);
    return c;
}

template <typename T>
std::shared_ptr<CsrHdMatrix<T>> PauliMatToCsrHdMatrix(std::shared_ptr<PauliMat<T>> a) {
    Index nnz = 0;
    auto &col = a->col_;
    auto &dim = a->dim_;
    auto &coeff = a->coeff_;
    auto &p = a->p_;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+ : nnz)), dim, static_cast<uint64_t>(1) << nQubitTh,
                     for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                         if (i <= col[i]) {
                             nnz++;
                         }
                     })
    Index *indptr = reinterpret_cast<Index *>(malloc(sizeof(Index) * (dim + 1)));
    Index *indices = reinterpret_cast<Index *>(malloc(sizeof(Index) * nnz));
    CTP<T> data = reinterpret_cast<CTP<T>>(malloc(sizeof(CT<T>) * nnz));
    indptr[0] = 0;
    for (Index i = 0, j = 0; i < dim; i++) {
        if (i <= col[i]) {
            indptr[i + 1] = indptr[i] + 1;
            if (i == col[i]) {
                data[j] = p * ComplexCast<double, T>::apply(POLAR[coeff[i]]) * (T) (0.5);
            } else {
                data[j] = p * ComplexCast<double, T>::apply(POLAR[coeff[i]]);
            }
            indices[j] = col[i];
            j++;
        } else {
            indptr[i + 1] = indptr[i];
        }
    }
    auto c = std::make_shared<CsrHdMatrix<T>>(dim, nnz, indptr, indices, data);
    return c;
}

template <typename T>
std::shared_ptr<PauliMat<T>> GetPauliMat(const PauliTerm<T> &pt, Index n_qubits) {
    auto c = std::make_shared<PauliMat<T>>(pt, n_qubits);
    return c;
}

template <typename T>
std::shared_ptr<CsrHdMatrix<T>> Csr_Plus_Csr(std::shared_ptr<CsrHdMatrix<T>> a, std::shared_ptr<CsrHdMatrix<T>> b) {
    auto a_nnz = a->nnz_;
    auto b_nnz = b->nnz_;
    auto dim = a->dim_;
    auto maxnnz = a_nnz + b_nnz;
    CTP<T> data = reinterpret_cast<CTP<T>>(malloc(sizeof(CT<T>) * maxnnz));
    Index *indices = reinterpret_cast<Index *>(malloc(sizeof(Index) * maxnnz));
    Index *indptr = reinterpret_cast<Index *>(malloc(sizeof(Index) * (dim + 1)));
    csr_plus_csr(dim, a->indptr_, a->indices_, a->data_, b->indptr_, b->indices_, b->data_, indptr, indices, data);
    auto nnz = indptr[dim];

    auto c = std::make_shared<CsrHdMatrix<T>>(dim, nnz, indptr, indices, data);
    return c;
}

template <typename T>
std::shared_ptr<CsrHdMatrix<T>> SparseHamiltonian(const VT<PauliTerm<T>> &hams, Index n_qubits) {
    VT<std::shared_ptr<CsrHdMatrix<T>>> sp_hams(hams.size());

    THRESHOLD_OMP_FOR(
        n_qubits, nQubitTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(hams.size()); i++) {
            auto pm = GetPauliMat(hams[i], n_qubits);
            sp_hams[i] = PauliMatToCsrHdMatrix(pm);
            pm->Reset();
        })
    Index tot = static_cast<Index>(hams.size());
    while (tot > 1) {
        Index half = tot / 2 + tot % 2;
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for schedule(static) num_threads(half)), n_qubits, nQubitTh,
                         for (omp::idx_t i = static_cast<omp::idx_t>(half); i < tot; i++) {
                             sp_hams[i - half] = Csr_Plus_Csr(sp_hams[i - half], sp_hams[i]);
                             sp_hams[i]->Reset();
                         })
        tot = half;
    }
    return sp_hams[0];
}

template <typename T, typename T2>
T2 *Csr_Dot_Vec(std::shared_ptr<CsrHdMatrix<T>> a, T2 *vec) {
    auto dim = a->dim_;
    auto c_vec = reinterpret_cast<CTP<T2>>(vec);
    auto new_vec = reinterpret_cast<CTP<T2>>(malloc(sizeof(CT<T2>) * dim));
    auto data = a->data_;
    auto indptr = a->indptr_;
    auto indices = a->indices_;
    THRESHOLD_OMP_FOR(
        dim, static_cast<uint64_t>(1) << nQubitTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            CT<T2> sum = {0.0, 0.0};
            for (omp::idx_t j = indptr[i]; j < static_cast<omp::idx_t>(indptr[i + 1]); j++) {
                sum += data[j] * c_vec[indices[j]];
            }
            new_vec[i] = sum;
        })
    return reinterpret_cast<T2 *>(new_vec);
}

template <typename T, typename T2>
CT<T2> ExpectationOfCsr(std::shared_ptr<CsrHdMatrix<T>> a, T2 *bra, T2 *ket) {
    auto dim = a->dim_;
    auto c_bra = reinterpret_cast<CTP<T2>>(bra);
    auto c_ket = reinterpret_cast<CTP<T2>>(ket);
    auto data = a->data_;
    auto indptr = a->indptr_;
    auto indices = a->indices_;
    T2 res_real = 0, res_imag = 0;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim,
                                                static_cast<uint64_t>(1) << nQubitTh,
                                                for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                                                    CT<T2> sum = {0.0, 0.0};
                                                    for (Index j = indptr[i]; j < indptr[i + 1]; j++) {
                                                        sum += data[j] * c_ket[indices[j]];
                                                    }
                                                    auto tmp = std::conj(c_bra[i]) * sum;
                                                    res_real += std::real(tmp);
                                                    res_imag += std::imag(tmp);
                                                })
    return {res_real, res_imag};
}

template <typename T, typename T2>
T2 *Csr_Dot_Vec(std::shared_ptr<CsrHdMatrix<T>> a, std::shared_ptr<CsrHdMatrix<T>> b, T2 *vec) {
    auto dim = a->dim_;
    auto c_vec = reinterpret_cast<CTP<T2>>(vec);
    auto new_vec = reinterpret_cast<CTP<T2>>(malloc(sizeof(CT<T2>) * dim));
    auto data = a->data_;
    auto indptr = a->indptr_;
    auto indices = a->indices_;
    auto data_b = b->data_;
    auto indptr_b = b->indptr_;
    auto indices_b = b->indices_;

    THRESHOLD_OMP_FOR(
        dim, static_cast<uint64_t>(1) << nQubitTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            CT<T2> sum = {0.0, 0.0};
            for (omp::idx_t j = indptr[i]; j < static_cast<omp::idx_t>(indptr[i + 1]); j++) {
                sum += data[j] * c_vec[indices[j]];
            }
            for (omp::idx_t j = indptr_b[i]; j < static_cast<omp::idx_t>(indptr_b[i + 1]); j++) {
                sum += data_b[j] * c_vec[indices_b[j]];
            }
            new_vec[i] = sum;
        })
    return reinterpret_cast<T2 *>(new_vec);
}

template <typename T, typename T2>
CT<T2> ExpectationOfCsr(std::shared_ptr<CsrHdMatrix<T>> a, std::shared_ptr<CsrHdMatrix<T>> b, T2 *bra, T2 *ket) {
    auto dim = a->dim_;
    auto c_bra = reinterpret_cast<CTP<T2>>(bra);
    auto c_ket = reinterpret_cast<CTP<T2>>(ket);
    auto data = a->data_;
    auto indptr = a->indptr_;
    auto indices = a->indices_;
    auto data_b = b->data_;
    auto indptr_b = b->indptr_;
    auto indices_b = b->indices_;
    T2 res_real = 0, res_imag = 0;

    // clang-format off
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim,
            static_cast<uint64_t>(1) << nQubitTh,
            for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                CT<T2> sum = {0.0, 0.0};
                for (omp::idx_t j = indptr[i]; j < static_cast<omp::idx_t>(indptr[i + 1]); j++) {
                    sum += data[j] * c_ket[indices[j]];
                }
                for (omp::idx_t j = indptr_b[i]; j < static_cast<omp::idx_t>(indptr_b[i + 1]); j++) {
                    sum += data_b[j] * c_ket[indices_b[j]];
                }
                auto tmp = std::conj(c_bra[i]) * sum;
                res_real += std::real(tmp);
                res_imag += std::imag(tmp);
            })
    // clang-format on
    return {res_real, res_imag};
}
}  // namespace mindquantum::sparse
#endif  // MINDQUANTUM_SPARSE_ALGO_H_
