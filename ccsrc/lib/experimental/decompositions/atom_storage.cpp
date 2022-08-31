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

#include "experimental/decompositions/atom_storage.hpp"

#include <algorithm>
#include <string_view>

namespace mindquantum::decompositions {
// =========================================================================
// :: get_atom_for

auto AtomStorage::get_atom_for(const instruction_t& inst) noexcept -> atom_t* {
    // NB: search backwards so that we get the most specialized decomposition atoms first (ie. more constrained on
    //     control qubits)
    if (auto it = std::find_if(rbegin(atoms_), rend(atoms_),
                               [&inst](const auto& atom) { return atom.second.is_applicable(inst); });
        it != std::rend(atoms_)) {
        return &it->second;
    }
    return nullptr;
}
}  // namespace mindquantum::decompositions
