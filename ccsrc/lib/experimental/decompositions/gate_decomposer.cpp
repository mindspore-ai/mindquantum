//   Copyright 2021 <Huawei Technologies Co., Ltd>
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

#include "experimental/decompositions/gate_decomposer.hpp"

#include <algorithm>

namespace mindquantum::decompositions {
// =========================================================================
// :: get_atom_for

auto GateDecomposer::get_atom_for(const instruction_t& inst) noexcept -> atom_t* {
    if (auto* atom = atom_storage_.get_atom_for(inst); atom != nullptr) {
        return atom;
    }

    if (auto it = std::find_if(begin(general_rule_storage_), end(general_rule_storage_),
                               [&inst](const auto& atom) { return atom.is_applicable(inst); });
        it != std::end(general_rule_storage_)) {
        /* NB: const_cast() is needed for compilers/STL implementations where std::set elements are always
         *     immutable when accessed through iterators.
         */
        return &const_cast<DecompositionAtom&>(*it);
    }

    return nullptr;
}
}  // namespace mindquantum::decompositions

// =============================================================================
