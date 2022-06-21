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

#ifndef PARTIAL_PLACER_HPP
#define PARTIAL_PLACER_HPP

#include <vector>

#include <tweedledum/IR/Instruction.h>
#include <tweedledum/IR/Qubit.h>
#include <tweedledum/Target/Device.h>
#include <tweedledum/Target/Placement.h>

namespace mindquantum::mapping {
class PartialPlacer {
 public:
    using qubit_t = tweedledum::Qubit;
    using device_t = tweedledum::Device;
    using placement_t = tweedledum::Placement;

    //! Constructor
    PartialPlacer(const device_t& device, placement_t& placement);

    //! Execute placing algorithm
    /*!
     * \note This placer should only be run **before** any operations have been added to the mapped circuit.
     */
    void run(const std::vector<qubit_t>& new_qubits);

 private:
    const device_t& device_;
    placement_t& placement_;
};
}  // namespace mindquantum::mapping

#endif /* PARTIAL_PLACER_HPP */
