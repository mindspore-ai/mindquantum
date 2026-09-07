/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include "config/openmp.h"
#include "core/mq_base_types.h"
#include "core/utils.h"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.h"
#include "simulator/state_index.h"

namespace mindquantum::sim::densitymatrix::detail {
namespace {
[[nodiscard]] auto QubitCount(index_t dim) -> qbit_t {
    if (dim == 0 || (dim & (dim - 1)) != 0) {
        throw std::invalid_argument("Density matrix dimension must be a non-zero power of two.");
    }
    qbit_t count = 0;
    while (dim > 1) {
        dim >>= 1;
        ++count;
    }
    return count;
}
}  // namespace

template <typename calc_type_>
auto CPUDensityMatrixPolicy<calc_type_>::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    if (dim == 0 || Size(dim) / dim != dim) {
        throw std::invalid_argument("Density matrix dimension is too large.");
    }
    auto* qs = static_cast<qs_data_p_t>(std::calloc(Size(dim), sizeof(qs_data_t)));
    if (qs == nullptr) {
        throw std::runtime_error("Allocate memory for quantum state failed.");
    }
    if (zero_state) {
        qs[0] = 1;
    }
    return qs;
}

template <typename calc_type_>
void CPUDensityMatrixPolicy<calc_type_>::FreeState(qs_data_p_t qs) noexcept {
    std::free(qs);
}

template <typename calc_type_>
auto CPUDensityMatrixPolicy<calc_type_>::Copy(const qs_data_p_t& qs, index_t dim) -> qs_data_p_t {
    auto* out = InitState(dim, false);
    CopyInto(out, qs, dim);
    return out;
}

template <typename calc_type_>
void CPUDensityMatrixPolicy<calc_type_>::CopyInto(qs_data_p_t des, const qs_data_p_t& src, index_t dim) {
    const auto size = Size(dim);
    MQ_OMP_PARALLEL_FOR(size, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(size); i++) { des[i] = src[i]; })
}

template <typename calc_type_>
void CPUDensityMatrixPolicy<calc_type_>::SetZeroState(qs_data_p_t qs, index_t dim) {
    const auto size = Size(dim);
    MQ_OMP_PARALLEL_FOR(size, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(size); i++) { qs[i] = 0; })
    qs[0] = 1;
}

template <typename calc_type_>
void CPUDensityMatrixPolicy<calc_type_>::Display(const qs_data_p_t& qs, qbit_t n_qubits, qbit_t q_limit) {
    const auto dim = index_t{1} << n_qubits;
    const auto shown = index_t{1} << std::min(n_qubits, q_limit);
    std::cout << n_qubits << " qubits cpu density matrix simulator (little endian)." << std::endl;
    for (index_t i = 0; i < shown; i++) {
        for (index_t j = 0; j < shown; j++) {
            const auto& value = qs[i * dim + j];
            std::cout << "(" << value.real() << ", " << value.imag() << "),";
        }
        std::cout << std::endl;
    }
}

template <typename calc_type_>
auto CPUDensityMatrixPolicy<calc_type_>::GetQS(const qs_data_p_t& qs, index_t dim) -> matrix_t {
    matrix_t out(dim, py_qs_datas_t(dim));
    MQ_OMP_PARALLEL_FOR(
        Size(dim), for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            std::copy_n(qs + static_cast<index_t>(i) * dim, dim, out[i].begin());
        })
    return out;
}

template <typename calc_type_>
void CPUDensityMatrixPolicy<calc_type_>::SetQS(qs_data_p_t qs, const py_qs_datas_t& state_vector, index_t dim) {
    if (state_vector.size() != dim) {
        throw std::invalid_argument(
            fmt::format("State vector size {} does not match the simulator dimension {}.", state_vector.size(), dim));
    }
    MQ_OMP_PARALLEL_FOR(
        Size(dim), for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            auto* row = qs + static_cast<index_t>(i) * dim;
            const auto amplitude = state_vector[i];
            for (index_t j = 0; j < dim; j++) {
                row[j] = amplitude * std::conj(state_vector[j]);
            }
        })
}

template <typename calc_type_>
void CPUDensityMatrixPolicy<calc_type_>::SetDM(qs_data_p_t qs, const matrix_t& density_matrix, index_t dim) {
    if (density_matrix.size() != dim
        || std::any_of(density_matrix.begin(), density_matrix.end(),
                       [dim](const auto& row) { return row.size() != dim; })) {
        throw std::invalid_argument(
            fmt::format("Density matrix shape does not match the simulator dimension {}.", dim));
    }
    MQ_OMP_PARALLEL_FOR(
        Size(dim), for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            std::copy_n(density_matrix[i].begin(), dim, qs + static_cast<index_t>(i) * dim);
        })
}

template <typename calc_type_>
auto CPUDensityMatrixPolicy<calc_type_>::Purity(const qs_data_p_t& qs, index_t dim) -> calc_type {
    const auto size = Size(dim);
    calc_type purity = 0;
    MQ_OMP_PARALLEL_FOR_CLAUSES(
        size, schedule(static) reduction(+ : purity),
        for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(size); i++) { purity += std::norm(qs[i]); })
    return purity;
}

template <typename calc_type_>
auto CPUDensityMatrixPolicy<calc_type_>::ReducedDensityMatrix(const qs_data_p_t& qs, const qbits_t& kept_qubits,
                                                              index_t dim) -> matrix_t {
    const auto n_qubits = QubitCount(dim);
    std::vector<bool> kept(static_cast<size_t>(n_qubits), false);
    for (size_t index = 0; index < kept_qubits.size(); ++index) {
        const auto qubit = kept_qubits[index];
        if (qubit < 0 || qubit >= n_qubits || (index != 0 && qubit <= kept_qubits[index - 1])) {
            throw std::invalid_argument("Kept qubits must be strictly ascending and within the simulator.");
        }
        kept[static_cast<size_t>(qubit)] = true;
    }
    qbits_t traced_qubits;
    for (qbit_t qubit = 0; qubit < n_qubits; ++qubit) {
        if (!kept[static_cast<size_t>(qubit)]) {
            traced_qubits.push_back(qubit);
        }
    }

    // Inserting zeros at the traced positions places a kept-subsystem index
    // into the full index, and vice versa.
    const state_index::BitExpander place_kept(traced_qubits);
    const state_index::BitExpander place_traced(kept_qubits);
    const auto kept_dim = index_t{1} << kept_qubits.size();
    const auto traced_dim = index_t{1} << traced_qubits.size();
    matrix_t out(kept_dim, py_qs_datas_t(kept_dim));
    MQ_OMP_PARALLEL_FOR(
        Size(dim), for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(kept_dim); a++) {
            const auto row_base = place_kept.Expand(static_cast<index_t>(a));
            for (index_t b = 0; b < kept_dim; b++) {
                const auto column_base = place_kept.Expand(b);
                qs_data_t sum = 0;
                for (index_t t = 0; t < traced_dim; t++) {
                    const auto traced = place_traced.Expand(t);
                    sum += qs[(row_base | traced) * dim + (column_base | traced)];
                }
                out[a][b] = sum;
            }
        })
    return out;
}

template <typename calc_type_>
auto CPUDensityMatrixPolicy<calc_type_>::PureStateVector(const qs_data_p_t& qs, index_t dim) -> py_qs_datas_t {
    if (1 - Purity(qs, dim) > 1e-6) {
        throw std::runtime_error("PureStateVector(): Cannot transform mixed density matrix to vector.");
    }
    index_t base = 0;
    while (base < dim && qs[base * dim + base].real() <= 1e-8) {
        ++base;
    }
    if (base == dim) {
        throw std::runtime_error("PureStateVector(): The density matrix has no populated basis state.");
    }
    // Every column of |psi><psi| is a multiple of |psi>.
    const auto base_amplitude = std::sqrt(qs[base * dim + base].real());
    py_qs_datas_t state_vector(dim);
    MQ_OMP_PARALLEL_FOR(
        Size(dim), for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            state_vector[i] = qs[static_cast<index_t>(i) * dim + base] / base_amplitude;
        })
    return state_vector;
}

template <typename calc_type_>
auto CPUDensityMatrixPolicy<calc_type_>::InnerProduct(const qs_data_p_t& bra, const qs_data_p_t& ket, index_t dim)
    -> py_qs_data_t {
    const auto size = Size(dim);
    calc_type real = 0;
    calc_type imag = 0;
    MQ_OMP_PARALLEL_FOR_CLAUSES(
        size, schedule(static) reduction(+ : real, imag),
        for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(size); i++) {
            const auto product = std::conj(bra[i]) * ket[i];
            real += product.real();
            imag += product.imag();
        })
    return {real, imag};
}

template struct CPUDensityMatrixPolicy<float>;
template struct CPUDensityMatrixPolicy<double>;
}  // namespace mindquantum::sim::densitymatrix::detail
