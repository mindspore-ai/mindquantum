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
#ifndef MINDQUANTUM_BACKENDS_PROJECTQ_UTILS_H_
#define MINDQUANTUM_BACKENDS_PROJECTQ_UTILS_H_
#include <utility>
#include <vector>

#include "core/utils.h"
#include "projectq/backends/_sim/_cppkernels/fusion.hpp"
#include "projectq/backends/_sim/_cppkernels/intrin/alignedallocator.hpp"
#include "projectq/backends/_sim/_cppkernels/simulator.hpp"

namespace mindquantum {
namespace projectq {
inline VT<unsigned> VCast(const VT<Index> &a) {
    return VT<unsigned>(a.begin(), a.end());
}

template <typename T>
inline Fusion::Matrix MCast(const VVT<CT<T>> &m) {
    Fusion::Matrix out;
    for (size_t i = 0; i < m.size(); i++) {
        std::vector<Fusion::Complex, aligned_allocator<Fusion::Complex, 64>> col;
        for (auto &a : m[i]) {
            // cppcheck-suppress useStlAlgorithm
            col.push_back({a.real(), a.imag()});
        }
        out.push_back(col);
    }
    return out;
}

template <typename T>
inline Simulator::ComplexTermsDict HCast(const VT<PauliTerm<T>> &ham_) {
    Simulator::ComplexTermsDict res;
    for (auto &pt : ham_) {
        Simulator::Term term;
        for (auto &pw : pt.first) {
            // cppcheck-suppress useStlAlgorithm
            term.push_back(std::make_pair(static_cast<unsigned>(pw.first), pw.second));
        }
        res.push_back(std::make_pair(term, static_cast<double>(pt.second)));
    }
    return res;
}
}  // namespace projectq
}  // namespace mindquantum
#endif  // MINDQUANTUM_BACKENDS_PROJECTQ_H_
