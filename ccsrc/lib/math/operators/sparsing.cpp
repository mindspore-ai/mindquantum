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
#include "math/operators/sparsing.h"

#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "core/mq_base_types.h"
#include "fmt/format.h"
#include "math/operators/qubit_operator_view.h"
#include "math/tensor/csr_matrix.h"
#include "math/tensor/ops_cpu/memory_operator.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"

namespace operators {
namespace mq = mindquantum;
template <typename T>
struct EleComp {
    using ele_t = std::pair<mq::index_t, T>;
    bool operator()(const ele_t& lhs, const ele_t& rhs) const {
        return lhs.first < rhs.first;
    }
};

template <typename T>
using ele_set = std::set<typename EleComp<T>::ele_t, EleComp<T>>;

template <tn::TDtype dtype, typename = std::enable_if<tn::is_complex_dtype_v<dtype>>>
tn::CsrMatrix GetMatrixImp(const qubit::QubitOperator& ops, int n_qubits) {
    if (tn::ToComplexType(ops.GetDtype()) != dtype) {
        throw std::invalid_argument(fmt::format("Data type mismatch for sparsing operator."));
    }
    using calc_type = tn::to_device_t<dtype>;
    mq::VT<calc_type> polar = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};

    auto n_local_qubits = ops.count_qubits();
    if (n_qubits < 0) {
        n_qubits = n_local_qubits;
    } else if (n_qubits < n_local_qubits) {
        throw std::runtime_error(
            fmt::format("n_qubits ({}) is less than local qubit number ({}).", n_qubits, n_local_qubits));
    }

    mq::VT<mq::PauliMask> pauli_mask;
    mq::VT<calc_type> consts;
    for (auto [term, pr] : ops.get_terms()) {
        consts.push_back(tn::ops::cpu::to_vector<calc_type>(pr.const_value)[0]);
        mq::VT<mq::index_t> out = {0, 0, 0, 0, 0, 0};
        for (auto& [idx, w] : term) {
            for (mq::index_t i = 0; i < 3; i++) {
                if (static_cast<mq::index_t>(w) - 1 == i) {
                    out[i] += (static_cast<uint64_t>(1) << idx);
                    out[3 + i] += 1;
                    break;
                }
            }
        }
        pauli_mask.push_back(mq::PauliMask({out[0], out[1], out[2], out[3], out[4], out[5]}));
    }

    auto dim = static_cast<uint64_t>(1) << n_qubits;
    mq::VT<ele_set<calc_type>> all_value(dim);
    mq::index_t nnz = 0;
#pragma omp parallel for schedule(static) reduction(+ : nnz)
    for (mq::index_t i = 0; i < dim; i++) {
        ele_set<calc_type> vals;
        for (mq::index_t term_idx = 0; term_idx < ops.size(); term_idx++) {
            auto& mask = pauli_mask[term_idx];
            auto mask_f = mask.mask_x | mask.mask_y;
            auto j = (i ^ mask_f);
            auto axis2power = mq::CountOne(static_cast<uint64_t>(i & mask.mask_z));  // -1
            auto axis3power = mq::CountOne(static_cast<uint64_t>(i & mask.mask_y));  // -1j
            auto val = polar[(mask.num_y + 2 * axis3power + 2 * axis2power) & 3] * consts[term_idx];
            typename EleComp<calc_type>::ele_t ele_new = {j, val};
            auto prev = vals.find(ele_new);
            if (prev != vals.end()) {
                ele_new.second += (*prev).second;
                vals.erase(prev);
            }
            if (std::abs(ele_new.second) < PRECISION) {
                continue;
            }
            vals.insert(ele_new);
        }
        nnz += vals.size();
        all_value[i] = vals;
    }
    auto indptr = reinterpret_cast<mq::index_t*>(malloc(sizeof(mq::index_t) * (dim + 1)));
    auto indices = reinterpret_cast<mq::index_t*>(malloc(sizeof(mq::index_t) * nnz));
    auto data = reinterpret_cast<calc_type*>(malloc(sizeof(calc_type) * nnz));
    indptr[0] = 0;
    for (mq::index_t i = 0; i < all_value.size(); i++) {
        auto& vals = all_value[i];
        indptr[i + 1] = indptr[i] + vals.size();
    }
#pragma omp parallel for schedule(static)
    for (mq::index_t i = 0; i < all_value.size(); i++) {
        auto& vals = all_value[i];
        auto begin = indptr[i];
        for (auto& [idx, val] : vals) {
            indices[begin] = idx;
            data[begin] = val;
            begin += 1;
        }
    }
    return tn::CsrMatrix(dim, dim, nnz, indptr, indices,
                         tn::Tensor(dtype, tn::TDevice::CPU, reinterpret_cast<void*>(data), nnz));
}

tn::CsrMatrix GetMatrix(const qubit::QubitOperator& ops, int n_qubits) {
    auto dtype = ops.GetDtype();
    tn::CsrMatrix out;
    switch (dtype) {
        case (tn::TDtype::Float32):
        case (tn::TDtype::Complex64):
            out = GetMatrixImp<tn::TDtype::Complex64>(ops, n_qubits);
            break;
        case (tn::TDtype::Float64):
        case (tn::TDtype::Complex128):
            out = GetMatrixImp<tn::TDtype::Complex128>(ops, n_qubits);
            break;
    }
    return out;
}

}  // namespace operators
