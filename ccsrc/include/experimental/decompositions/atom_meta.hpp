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

#ifndef ATOM_META_HPP
#define ATOM_META_HPP

#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/parametric_atom.hpp"
#include "experimental/decompositions/trivial_atom.hpp"

namespace mindquantum::traits {
#if MQ_HAS_CONCEPTS
template <typename T, decompositions::num_control_t num_controls>
struct atom_control_type;
template <concepts::NonParametricGate gate_t, decompositions::num_control_t num_controls>
struct atom_control_type<gate_t, num_controls> {
    using control_type = decompositions::TrivialSimpleAtom<gate_t, num_controls>;
};
template <concepts::ParametricGate gate_t, decompositions::num_control_t num_controls>
struct atom_control_type<gate_t, num_controls> {
    using control_type = decompositions::ParametricSimpleAtom<gate_t, num_controls>;
};
#else
template <typename gate_t, decompositions::num_control_t num_controls, typename = void>
struct atom_control_type {
    using control_type = decompositions::TrivialSimpleAtom<gate_t, num_controls>;
};
template <typename gate_t, decompositions::num_control_t num_controls>
struct atom_control_type<gate_t, num_controls, typename std::void_t<typename gate_t::is_parametric>> {
    using control_type = decompositions::ParametricSimpleAtom<gate_t, num_controls>;
};
#endif  // MQ_HAS_CONCEPTS
}  // namespace mindquantum::traits

namespace mindquantum::decompositions::atoms {
template <typename op_t, num_control_t num_controls = num_control_t(1L)>
using Control = typename traits::atom_control_type<op_t, num_controls>::control_type;
template <typename op_t, num_control_t num_controls = num_control_t(1L)>
using C = Control<op_t, num_controls>;
}  // namespace mindquantum::decompositions::atoms

#endif /* ATOM_META_HPP */
