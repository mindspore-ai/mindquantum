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

#ifndef DECOMPOSITION_ATOM_HPP
#define DECOMPOSITION_ATOM_HPP

#include <algorithm>
#include <array>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "experimental/core/config.hpp"
#include "experimental/core/control.hpp"
#include "experimental/core/operator_traits.hpp"
#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/details/concepts.hpp"
#include "experimental/ops/gates/invalid.hpp"
#include "experimental/ops/parametric/config.hpp"
#include "experimental/ops/parametric/register_gate_type.hpp"

namespace mindquantum::decompositions {
class DecompositionAtom {
 public:
    using gate_param_t = mindquantum::ops::parametric::gate_param_t;

    template <typename atom_t,
              typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<atom_t>, DecompositionAtom>>>
    DecompositionAtom(atom_t&& atom) noexcept {  // NOLINT(runtime/explicit)
        static_assert(std::is_trivially_copyable_v<std::remove_cvref_t<atom_t>>);
        constexpr bool is_small = sizeof(Model<std::remove_cvref_t<atom_t>, true>) <= small_size;
        new (&model_) Model<std::remove_cvref_t<atom_t>, is_small>(std::forward<atom_t>(atom));
        concept_ = &Model<std::remove_cvref_t<atom_t>, is_small>::vtable_;
    }

    DecompositionAtom(const DecompositionAtom& other) noexcept : concept_(other.concept_) {
        concept_->clone(&other.model_, &model_);
    }
    DecompositionAtom& operator=(const DecompositionAtom& other) noexcept {
        if (this != &other) {
            concept_->dtor(&model_);

            concept_ = other.concept_;
            concept_->clone(&other.model_, &model_);
        }
        return *this;
    }

    DecompositionAtom(DecompositionAtom&& other) noexcept : concept_(other.concept_) {
        std::copy(begin(other.model_), end(other.model_), begin(model_));
        std::fill(begin(other.model_), end(other.model_), std::byte{0});
        other.concept_ = nullptr;
    }
    DecompositionAtom& operator=(DecompositionAtom&& other) noexcept {
        if (this != &other) {
            concept_ = other.concept_;
            std::copy(begin(other.model_), end(other.model_), begin(model_));
            std::fill(begin(other.model_), end(other.model_), std::byte{0});
            other.concept_ = nullptr;
        }
        return *this;
    }

    ~DecompositionAtom() noexcept {
        if (concept_ != nullptr) {
            concept_->dtor(&model_);
        }
    }

    //! Return the name of this decomposition atom
    MQ_NODISCARD auto name() const noexcept {
        return concept_->name();
    }

    //! Test whether an atom has (supports) a particular kind of operator
    MQ_NODISCARD auto is_kind(std::string_view kind) const noexcept {
        return concept_->is_kind(kind);
    }

    //! Test whether this atom is applicable to a particular instruction
    /*!
     * Child classes may implement a method named \c is_applicable_impl in order to customize the default
     * behaviour for this method.
     *
     * \param inst An instruction
     * \return True if the atom can be applied, false otherwise
     */
    MQ_NODISCARD bool is_applicable(const instruction_t& inst) const noexcept {
        return concept_->is_applicable(&model_, inst);
    }

    //! Apply a decomposition atom to decompose an instruction
    /*!
     * \pre is_applicable() returns true
     * \param circuit A quantum circuit to apply the decomposition atom to
     * \param inst A quantum instruction to decompose
     */
    void apply(circuit_t& circuit, const instruction_t& inst) noexcept {
        return concept_->apply(&model_, circuit, inst);
    }

    //! Apply the atom (ie. the decomposition it represents) to a quantum circuit
    /*!
     * This overload assumes the decomposition atom is not parametric
     *
     * \param circuit A quantum circuit to apply the decomposition atom to
     * \param op A quantum operation
     * \param qubits A list of qubits to apply the decomposition atom
     * \param cbits A list of classical bit the decomposition applies to
     *
     * \note Currently the \c cbits parameter is not used at all! It is here to make the API futureproof.
     */
    void apply(circuit_t& circuit, const operator_t& op, const qubits_t& qubits, const cbits_t& cbits = {}) noexcept {
        concept_->apply_operator(&model_, circuit, op, qubits, cbits);
    }

 private:
    struct Concept {
        void (*dtor)(void*) noexcept;  // NOLINT(readability/casting)
        void (*clone)(void const*, void*) noexcept;
        std::string_view (*name)() noexcept;
        bool (*is_kind)(std::string_view) noexcept;
        bool (*is_applicable)(void const*, const instruction_t&) noexcept;
        void (*apply)(void*, circuit_t&, const instruction_t&) noexcept;
        void (*apply_operator)(void*, circuit_t&, const operator_t&, const qubits_t&, const cbits_t&) noexcept;
    } MQ_ALIGN(64);

    template <class ConcreteOp, bool IsSmall>
    struct Model;

    static constexpr size_t small_size = sizeof(void*) * 4;

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    alignas(64) std::array<std::byte, small_size> model_{};
    Concept const* concept_;
};

namespace details {
template <typename atom_t>
void apply_gate(atom_t& atom, circuit_t& circuit, const instruction_t& inst) {
    /* NB: The approach below has one key limitation in that it is not able to handle decomposition rules that
     *     match the following conditions:
     *       - non-constant number of target qubits
     *       - no constraints on the number of control qubits
     *       - the decomposition needs access to some of the control qubits
     *
     *     e.g. imagine you have a quantum operator `A` that has variable number of targets and you would like
     * to write a decomposition rule that should apply regardless of the number of control qubits and that aim
     * to reduce the number of control qubits by 1, such that:
     *            - CCCA -> CA + some other gates
     *            - CCA -> CA + some other gates
     *            - CA -> A + some other gates
     *
     *          As it is now, this decomposition rule will have num_controls_for_decomp == 0 and therefore never
     *          receive any control qubits when its apply(...) methods are called.
     *
     *      However, if the number of target qubits is fixed, no such limitation exists.
     */

    assert(inst.num_controls() >= atom_t::num_controls_for_decomp);
    const auto qubit_offset = inst.num_controls() - atom_t::num_controls_for_decomp;

    auto free_controls = qubits_t{};
    for (auto i(0); i < qubit_offset; ++i) {
        free_controls.emplace_back(inst.qubit(i));
    }
    auto qubits = qubits_t{};
    for (auto i(qubit_offset); i < inst.num_qubits(); ++i) {
        qubits.emplace_back(inst.qubit(i));
    }

    MQ_WITH_CONTROL(circuit, controlled, free_controls) {
        // TODO(dnguyen): Fix cbits argument if required in the future!
        atom.apply(controlled, static_cast<const operator_t&>(inst), qubits, {});
    }
}
}  // namespace details

// Stack
template <class atom_t>
struct DecompositionAtom::Model<atom_t, true> {
    using type = atom_t;

    explicit Model(atom_t&& op) noexcept : operator_(std::forward<atom_t>(op)) {
    }

    explicit Model(atom_t const& op) noexcept : operator_(op) {
    }

    static auto* self_cast(void* self) noexcept {
        return std::launder(reinterpret_cast<Model*>(self));
    }
    static auto* self_cast(const void* self) noexcept {
        return std::launder(reinterpret_cast<const Model*>(self));
    }

    static void dtor(void* self) noexcept {
        self_cast(self)->~Model();
    }

    static void clone(void const* self, void* other) noexcept {
        new (other) Model<std::decay_t<atom_t>, true>(self_cast(self)->operator_);
    }

    static constexpr std::string_view name() noexcept {
        return atom_t::name();
    }

    static constexpr bool is_kind(std::string_view kind) noexcept {
        if constexpr (concepts::GateDecomposition<atom_t>) {
            return traits::kind_compare<atom_t>(kind);
        } else {
            return false;
        }
    }

    static bool is_applicable(void const* self, const instruction_t& inst) noexcept {
        return self_cast(self)->operator_.is_applicable(inst);
    }

    static void apply(void* self, circuit_t& circuit, const instruction_t& inst) noexcept {
        if constexpr (concepts::GateDecomposition<atom_t>) {
            details::apply_gate(self_cast(self)->operator_, circuit, inst);
        } else {
            self_cast(self)->operator_.apply(circuit, inst);
        }
    }

    static void apply_operator(void* self, circuit_t& circuit, const operator_t& op, const qubits_t& qubits,
                               const cbits_t& cbits) noexcept {
        if constexpr (concepts::GateDecomposition<atom_t>) {
            self_cast(self)->operator_.apply(circuit, op, qubits, cbits);
        } else {
            circuit.apply_operator(ops::Invalid{}, qubits);
        }
    }

    static constexpr Concept vtable_{dtor, clone, name, is_kind, is_applicable, apply, apply_operator};

    atom_t operator_;
};

// Heap
template <class atom_t>
struct DecompositionAtom::Model<atom_t, false> {
    explicit Model(atom_t&& op) noexcept : operator_(std::make_unique<atom_t>(std::forward<atom_t>(op))) {
    }

    explicit Model(atom_t const& op) noexcept : operator_(std::make_unique<atom_t>(op)) {
    }

    static auto* self_cast(void* self) noexcept {
        return std::launder(reinterpret_cast<Model*>(self));
    }
    static auto* self_cast(const void* self) noexcept {
        return std::launder(reinterpret_cast<const Model*>(self));
    }

    static void dtor(void* self) noexcept {
        self_cast(self)->~Model();
    }

    static void clone(void const* self, void* other) noexcept {
        new (other) Model<atom_t, false>(*self_cast(self)->operator_);
    }

    static constexpr std::string_view name() noexcept {
        return atom_t::name();
    }

    static constexpr bool is_kind(std::string_view kind) noexcept {
        if constexpr (concepts::GateDecomposition<atom_t>) {
            return traits::kind_compare<atom_t>(kind);
        } else {
            return false;
        }
    }

    static bool is_applicable(void const* self, const instruction_t& inst) noexcept {
        return self_cast(self)->operator_->is_applicable(inst);
    }

    static void apply(void* self, circuit_t& circuit, const instruction_t& inst) noexcept {
        if constexpr (concepts::GateDecomposition<atom_t>) {
            details::apply_gate(*(self_cast(self)->operator_), circuit, inst);
        } else {
            self_cast(self)->operator_->apply(circuit, inst);
        }
    }

    static void apply_operator(void* self, circuit_t& circuit, const operator_t& op, const qubits_t& qubits,
                               const cbits_t& cbits) noexcept {
        if constexpr (concepts::GateDecomposition<atom_t>) {
            self_cast(self)->operator_->apply(circuit, op, qubits, cbits);
        } else {
            circuit.apply_operator(ops::Invalid{std::size(qubits)}, qubits);
        }
    }

    static constexpr Concept vtable_{dtor, clone, name, is_kind, is_applicable, apply, apply_operator};

    std::unique_ptr<atom_t> operator_;
};
}  // namespace mindquantum::decompositions

#endif /* DECOMPOSITION_ATOM_HPP */
