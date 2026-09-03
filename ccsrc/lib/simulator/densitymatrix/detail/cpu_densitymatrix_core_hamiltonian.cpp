/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include <memory>
#include <stdexcept>
#include <vector>

#include "config/openmp.h"
#include "config/type_promotion.h"
#include "core/mq_base_types.h"
#include "core/sparse/algo.h"
#include "core/sparse/csrhdmatrix.h"
#include "core/utils.h"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.h"
#include "simulator/utils.h"

namespace mindquantum::sim::densitymatrix::detail {
namespace {
// A Pauli string P maps |i> to Phase(i) |i ^ flip>.
template <typename calc_type>
struct PauliAction {
    explicit PauliAction(const std::vector<PauliWord>& words) : mask(GenPauliMask(words)) {
    }

    [[nodiscard]] auto Phase(index_t basis) const -> std::complex<calc_type> {
        const auto quarter_turns = mask.num_y + 2 * CountOne(static_cast<std::uint64_t>(basis & mask.mask_y))
                                   + 2 * CountOne(static_cast<std::uint64_t>(basis & mask.mask_z));
        return ComplexCast<double, calc_type>::apply(POLAR[static_cast<char>(quarter_turns & 3)]);
    }

    [[nodiscard]] auto Flip() const -> index_t {
        return mask.mask_x | mask.mask_y;
    }

    PauliMask mask;
};

template <typename calc_type>
[[nodiscard]] auto SparseMatrix(const Hamiltonian<calc_type>& ham, index_t dim)
    -> std::shared_ptr<sparse::CsrHdMatrix<calc_type>> {
    auto matrix = ham.how_to_ == BACKEND ? sparse::Csr_Plus_Csr<calc_type>(ham.ham_sparse_main_, ham.ham_sparse_second_)
                                         : ham.ham_sparse_main_;
    if (matrix == nullptr || matrix->dim_ != dim) {
        throw std::invalid_argument("Sparse Hamiltonian size does not match the quantum state size.");
    }
    return matrix;
}
}  // namespace

template <typename calc_type_>
auto CPUDensityMatrixPolicy<calc_type_>::HamiltonianMatrix(const Hamiltonian<calc_type>& ham, index_t dim)
    -> qs_data_p_t {
    auto* out = InitState(dim, false);
    if (ham.how_to_ == ORIGIN) {
        for (const auto& [words, coefficient] : ham.ham_) {
            const PauliAction<calc_type> pauli(words);
            const auto flip = pauli.Flip();
            const auto weight = coefficient;
            THRESHOLD_OMP_FOR(
                Size(dim), DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                    const auto column = static_cast<index_t>(i);
                    out[(column ^ flip) * dim + column] += weight * pauli.Phase(column);
                })
        }
        return out;
    }
    const auto matrix = SparseMatrix(ham, dim);
    THRESHOLD_OMP_FOR(
        Size(dim), DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            const auto row = static_cast<index_t>(i);
            for (auto k = matrix->indptr_[row]; k < matrix->indptr_[row + 1]; k++) {
                out[row * dim + matrix->indices_[k]] += matrix->data_[k];
            }
        })
    return out;
}

template <typename calc_type_>
auto CPUDensityMatrixPolicy<calc_type_>::Expectation(const qs_data_p_t& qs, const Hamiltonian<calc_type>& ham,
                                                     index_t dim) -> py_qs_data_t {
    calc_type real = 0;
    calc_type imag = 0;
    if (ham.how_to_ == ORIGIN) {
        // Tr(P rho) = sum_k Phase(k) rho[k][k ^ flip]
        for (const auto& [words, coefficient] : ham.ham_) {
            const PauliAction<calc_type> pauli(words);
            const auto flip = pauli.Flip();
            const auto weight = coefficient;
            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+ : real, imag)), Size(dim), DimTh,
                             for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                                 const auto row = static_cast<index_t>(i);
                                 const auto term = weight * pauli.Phase(row) * qs[row * dim + (row ^ flip)];
                                 real += term.real();
                                 imag += term.imag();
                             })
        }
        return {real, imag};
    }
    // Tr(H rho) = sum_i sum_k H[i][k] rho[k][i]
    const auto matrix = SparseMatrix(ham, dim);
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+ : real, imag)), Size(dim), DimTh,
                     for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                         const auto column = static_cast<index_t>(i);
                         qs_data_t sum = 0;
                         for (auto k = matrix->indptr_[column]; k < matrix->indptr_[column + 1]; k++) {
                             sum += matrix->data_[k] * qs[matrix->indices_[k] * dim + column];
                         }
                         real += sum.real();
                         imag += sum.imag();
                     })
    return {real, imag};
}

template <typename calc_type_>
void CPUDensityMatrixPolicy<calc_type_>::ApplyHamiltonian(qs_data_p_t qs, const Hamiltonian<calc_type>& ham,
                                                          index_t dim) {
    // H rho H^dagger in two passes: rows first, then columns.
    auto* left = InitState(dim, false);
    if (ham.how_to_ == ORIGIN) {
        for (const auto& [words, coefficient] : ham.ham_) {
            const PauliAction<calc_type> pauli(words);
            const auto flip = pauli.Flip();
            const auto weight = coefficient;
            THRESHOLD_OMP_FOR(
                Size(dim), DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                    const auto source = static_cast<index_t>(i);
                    const auto factor = weight * pauli.Phase(source);
                    auto* target_row = left + (source ^ flip) * dim;
                    const auto* source_row = qs + source * dim;
                    for (index_t j = 0; j < dim; j++) {
                        target_row[j] += factor * source_row[j];
                    }
                })
        }
        SetZeroState(qs, dim);
        qs[0] = 0;
        for (const auto& [words, coefficient] : ham.ham_) {
            const PauliAction<calc_type> pauli(words);
            const auto flip = pauli.Flip();
            const auto weight = coefficient;
            THRESHOLD_OMP_FOR(
                Size(dim), DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                    auto* target_row = qs + static_cast<index_t>(i) * dim;
                    const auto* source_row = left + static_cast<index_t>(i) * dim;
                    for (index_t j = 0; j < dim; j++) {
                        const auto source = j ^ flip;
                        target_row[j] += std::conj(weight * pauli.Phase(source)) * source_row[source];
                    }
                })
        }
        FreeState(left);
        return;
    }

    const auto matrix = SparseMatrix(ham, dim);
    THRESHOLD_OMP_FOR(
        Size(dim), DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            const auto row = static_cast<index_t>(i);
            auto* target_row = left + row * dim;
            for (auto k = matrix->indptr_[row]; k < matrix->indptr_[row + 1]; k++) {
                const auto factor = matrix->data_[k];
                const auto* source_row = qs + matrix->indices_[k] * dim;
                for (index_t j = 0; j < dim; j++) {
                    target_row[j] += factor * source_row[j];
                }
            }
        })
    THRESHOLD_OMP_FOR(
        Size(dim), DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            auto* target_row = qs + static_cast<index_t>(i) * dim;
            const auto* source_row = left + static_cast<index_t>(i) * dim;
            for (index_t j = 0; j < dim; j++) {
                qs_data_t sum = 0;
                for (auto k = matrix->indptr_[j]; k < matrix->indptr_[j + 1]; k++) {
                    sum += std::conj(matrix->data_[k]) * source_row[matrix->indices_[k]];
                }
                target_row[j] = sum;
            }
        })
    FreeState(left);
}

template struct CPUDensityMatrixPolicy<float>;
template struct CPUDensityMatrixPolicy<double>;
}  // namespace mindquantum::sim::densitymatrix::detail
