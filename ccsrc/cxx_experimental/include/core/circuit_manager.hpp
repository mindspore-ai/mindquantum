//   Copyright 2020 <Huawei Technologies Co., Ltd>
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

#ifndef CIRCUIT_MANAGER_HPP
#define CIRCUIT_MANAGER_HPP

#include "core/config.hpp"

#include "circuit_block.hpp"
#if MQ_HAS_CONCEPTS
#    include "engine_concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <tweedledum/IR/Circuit.h>
#include <tweedledum/Passes/Utility/shallow_duplicate.h>
#include <tweedledum/Target/Mapping.h>

#ifdef UNIT_TESTS
class UnitTestAccessor;
#endif  // UNIT_TESTS

namespace mindquantum {
namespace details {
class ViewBase;
class ExternalView;
class ExternalBlockView;
class PhysicalView;
}  // namespace details

static constexpr struct committed_t {
} committed;
static constexpr struct uncommitted_t {
} uncommitted;

//! Class keeping track of the mapping between various Qubit/Cbit ID types
/*!
 * There are mainly three sets of qubit IDs that we need to keep track of:
 *   - External IDs: assigned to qubits by External
 *   - Tweedledum IDs: ID of the wire in the underlying tweedledum network
 *   - Physical IDs: ID of the wire on the physical hardware
 *
 * In addition to the above three, mapping qubit IDs introduces an additional set of IDs:
 *   - virtual IDs: normalised set of IDs within the mapped dag
 *
 * However the latter type is not really relevant in most cases so these are not stored within a CircuitManager
 * instance.
 *
 * This class is also responsible for managing the internal circuit representation. The current representation
 * uses a list of CircuitBlock in order to represent the full circuit. The last element of that list is called
 * the "uncommitted" block in the following and is the circuit block to which all new operations get added.
 *
 * Upon certain triggering events, a new block may be added to the list in which case the current uncommitted
 * block becomes "committed" and a new fresh cicuit block gets added to the list.
 */
class CircuitManager {
 public:
    using circuit_t = CircuitBlock::circuit_t;
    using qubit_t = CircuitBlock::qubit_t;
    using inst_ref_t = CircuitBlock::inst_ref_t;
    using instruction_t = CircuitBlock::instruction_t;
    using placement_t = CircuitBlock::placement_t;
    using mapping_t = CircuitBlock::mapping_t;
    using device_t = CircuitBlock::device_t;
    using ext_id_t = CircuitBlock::ext_id_t;
    using td_qid_t = qubit_t;
    using td_cid_t = tweedledum::Cbit;
    using phys_id_t = qubit_t;

    CircuitManager();

    // ---------------------------

    //! Commit the changes of the current circuit
    /*!
     * Calls \c commit() on the current circuit, moves it to storage and add a new empty circuit for processing
     */
    void commit_changes();

    // ---------------------------

    //! Get the total number of (committed) operations currently managed
    std::size_t size(committed_t) const;

    //! Get the total number of (uncommitted) operations currently managed
    std::size_t size(uncommitted_t) const;

    //! Get the total number of operations currently managed
    std::size_t size() const;

    //! Check whether a mapping was performed
    /*!
     * \note Looks at the first block when making the determination.
     */
    bool has_mapping() const;

    //! Check whether a qubit with corresponding ID is known
    /*!
     * \note Check is performed on the last (uncommitted) block.
     */
    bool has_qubit(ext_id_t qubit_id) const;

    //! Convert internal (uncommitted) qubit IDs to External IDs
    /*!
     * \param ref Internal ID (Tweedledum wire reference)
     */
    qubit_id_t translate_id(const td_qid_t& ref) const;

    // ---------------------------

    //! Add a qubit to the circuit
    /*!
     * \param qubit_id (External) qubit ID to add
     * \return True if qubit could be added, false otherwise
     */
    bool add_qubit(ext_id_t qubit_id);

    //! Remove some qubits from the circuit
    /*!
     * \param qubits List of External qubit IDs of the qubits to remove
     */
    void delete_qubits(const std::vector<ext_id_t>& qubits);

    // ---------------------------

    //! Add an operation to the underlying network
    /*!
     * \param optor Operator to apply
     * \param control_qubits List of External qubit IDs representing controls
     * \param target_qubits List of External qubit IDs representing targets
     */
    template <typename OpT>
    inst_ref_t apply_operator(OpT&& optor, const qureg_t& control_qubits, const qureg_t& target_qubits);

    //! Add an operation to the underlying network
    /*!
     * \param optor Operator to apply
     * \param control_qubits List of External qubit IDs representing controls
     * \param target_qubits List of External qubit IDs representing targets
     */
    inst_ref_t apply_operator(const instruction_t& optor, const qureg_t& control_qubits, const qureg_t& target_qubits);

    //! Apply a measurement on a qubit
    /*!
     * \param id External qubit ID
     */
    inst_ref_t apply_measurement(qubit_id_t id);

    // ---------------------------

#if MQ_HAS_CONCEPTS
    template <concepts::Mapper mapper_t>
#else
    template <typename mapper_t>
#endif  // MQ_HAS_CONCEPTS
    void apply_mapping(const mapper_t& mapper);

    //! Apply a mapping to the underlying circuit
    /*!
     * \param device Hardware device the mapping is applied to
     * \param cold_start Callable to do a "cold start" mapping (ie. first mapping)
     * \param hot_start Callable to do a "hot start" mapping (ie. subsqeuent mapping)
     *
     * \note The signature of the callables are:
     *         - void (*) (td::MapState&);
     */
#if MQ_HAS_CONCEPTS
    template <concepts::cold_start_t Fn, concepts::hot_start_t Gn>
#else
    template <typename Fn, typename Gn>
#endif  // MQ_HAS_CONCEPTS
    void apply_mapping(const device_t& device, Fn&& cold_start, Gn&& hot_start);

    // ---------------------------

    //! Iterate over all committed blocks
    /*!
     * \param fn Callable to apply to each block.
     *
     * \warning This function does not do any ID re-mapping so a qubit might have different IDs between circuit
     *          blocks!
     *          Please use the as_XXX(...) methods if you want to access the qubit in a consistent way.
     */
    template <typename Fn>
    void foreach_block(const Fn& fn) const;

    //! Iterate over all committed blocks in reverse
    /*!
     * \param fn Callable to apply to each block.
     *
     * \warning This function does not do any ID re-mapping so a qubit might have different IDs between circuit
     *          blocks!
     *          Please use the as_XXX(...) methods if you want to access the qubit in a consistent way.
     */
    template <typename Fn>
    void foreach_r_block(const Fn& fn) const;

    //! Iterate over all uncommitted operations
    template <typename Fn>
    void foreach_instruction(Fn&& fn, uncommitted_t) const;

    //! Apply a transform to the (uncommitted) part of the circuit
    /*!
     * \tparam Fn Needs to be a callable type with one of the following
     *            signatures:
     *              - void (*) (const td::Circuit&);
     *              - td::Circuit (*) (const td::Circuit&);<br>
     *
     * \note The first signature does not modify the underlying network,
     *       while the second assigns the result of the transformation
     *       to the underlying network.
     */
    template <typename Fn>
    void transform(Fn&& fn);

    //! Return a view of the (committed) circuit with External IDs
    details::ExternalView as_projectq(committed_t) const;
    //! Return a view of the (uncommitted) circuit with External IDs
    details::ExternalBlockView as_projectq(uncommitted_t) const;

    //! Return a view of the (committed) circuit with physical/internal IDs
    /*!
     * \note In case no mapping was performed, \c PhysicalView will behave like a \c ExternalView (ie. using
     *      External IDs)
     */
    details::PhysicalView as_physical(committed_t) const;

 private:
#ifdef UNIT_TESTS
    friend class ::UnitTestAccessor;
#endif  // UNIT_TESTS

    std::vector<CircuitBlock> blocks_;
};
}  // namespace mindquantum

#include "circuit_manager.tpp"
#include "details/external_view.hpp"
#include "details/physical_view.hpp"

#endif /* CIRCUIT_MANAGER_HPP */
