//   Copyright 2022 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

#include <cmath>

#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <ratio>
#include <stdexcept>
#include <vector>

#include "config/openmp.hpp"

#include "core/utils.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"

namespace mindquantum::sim::densitymatrix::detail {

index_t Idx(index_t x, index_t y) {
    return (x * x + x) / 2 + y;
}

auto CPUDensityMatrixPolicyBase::InitState(index_t n_elements, bool zero_state) -> qs_data_p_t {
    auto qs = reinterpret_cast<qs_data_p_t>(calloc(n_elements, sizeof(qs_data_t)));
    if (zero_state) {
        qs[0] = 1;
    }
    return qs;
}

void CPUDensityMatrixPolicyBase::Reset(qs_data_p_t qs, index_t n_elements) {
    THRESHOLD_OMP_FOR(
        n_elements, DimTh, for (omp::idx_t i = 0; i < n_elements; i++) { qs[i] = 0; })
    qs[0] = 1;
}

void CPUDensityMatrixPolicyBase::FreeState(qs_data_p_t qs) {
    if (qs != nullptr) {
        free(qs);
    }
}

// need to fix
void CPUDensityMatrixPolicyBase::Display(qs_data_p_t qs, qbit_t n_qubits, qbit_t q_limit) {
    if (n_qubits > q_limit) {
        n_qubits = q_limit;
    }
    std::cout << n_qubits << " qubits cpu simulator (little endian)." << std::endl;
    for (index_t i = 0; i < (1UL << n_qubits); i++) {
        std::cout << "(" << qs[i].real() << ", " << qs[i].imag() << ")" << std::endl;
    }
}

auto CPUDensityMatrixPolicyBase::Copy(qs_data_p_t qs, index_t n_elements) -> qs_data_p_t {
    qs_data_p_t out = CPUDensityMatrixPolicyBase::InitState(n_elements, false);
    THRESHOLD_OMP_FOR(
        n_elements, DimTh, for (omp::idx_t i = 0; i < n_elements; i++) { out[i] = qs[i]; })
    return out;
}

auto CPUDensityMatrixPolicyBase::GetQS(qs_data_p_t qs, index_t dim) -> py_qs_datas_t {
    py_qs_datas_t out(dim, std::vector<py_qs_data_t>(dim));
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            for (index_t j = 0; j <= i; j++) {
                out[i][j] = qs[Idx(i, j)];
            }
            for (index_t j = i + 1; j < dim; j++) {
                out[i][j] = std::conj(qs[Idx(j, i)]);
            }
        })
    return out;
}

// can be imporved
bool CPUDensityMatrixPolicyBase::IsPure(qs_data_p_t qs, index_t dim, index_t n_elements) {
    auto qs_square = reinterpret_cast<qs_data_p_t>(calloc(n_elements, sizeof(qs_data_t)));
    index_t row = 0;
    index_t col = 0;
    for (index_t element = 0; element < n_elements; element++) {
        for (index_t i = 0; i < dim; i++) {
            qs_square[element] += qs[Idx(row, i)] * qs[Idx(i, col)];
        }
        if (qs_square[element] != qs[element]) {
            return false;
        }
        if (col == row) {
            row += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    return true;
}

// auto CPUDensityMatrixPolicyBase::MatrixMul(qs_data_p_t qs, )

void CPUDensityMatrixPolicyBase::DisplayQS(qs_data_p_t qs, qbit_t n_qubits, index_t dim) {
    auto out = CPUDensityMatrixPolicyBase::GetQS(qs, dim);
    std::cout << n_qubits << " qubits cpu simulator (little endian)." << std::endl;
    for (index_t i = 0; i < dim; i++) {
        for (index_t j = 0; j < dim; j++) {
            std::cout << "(" << out[i][j].real() << ", " << out[i][j].imag() << ")"
                      << ",";
        }
        std::cout << std::endl;
    }
}

// need to fix
void CPUDensityMatrixPolicyBase::SetQS(qs_data_p_t qs, const py_qs_datas_t& qs_out, index_t dim, index_t n_elements) {
    if (qs_out.size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { qs[i] = qs_out[i][i]; })
}
}  // namespace mindquantum::sim::densitymatrix::detail
