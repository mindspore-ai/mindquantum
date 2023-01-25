//   Copyright 2023 <Huawei Technologies Co., Ltd>
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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ops/basic_gate.hpp"
#include "ops/gate_id.hpp"
#include "ops/gates.hpp"
#include "simulator/vector/detail/cpu_vector_avx_double_policy.hpp"
#include "simulator/vector/runtime/rt_gate.h"
#include "simulator/vector/runtime/utils.h"
#include "simulator/vector/vector_state.hpp"

namespace mindquantum::sim::rt {
int cmd(const std::vector<std::string> &args) {
    if (args.size() < 4) {
        throw std::runtime_error("You should set n_qubits and random seed when running simulator.");
    }

    int n_qubits = std::get<1>(convert_int(args[2], MAX_QUBIT));
    int seed = std::get<1>(convert_int(args[3], MAX_SEED));
    int cmd_idx = 4;
    State state = State::W_GATE;
    std::vector<std::shared_ptr<BasicGate>> circ;
    Gate gate = Gate();
    int n_obj = 0;
    std::vector<std::string> gate_set1 = {"X", "Y", "Z", "H", "S", "T", "Sdag", "Tdag"};
    std::vector<std::string> gate_set2 = {"SWAP", "ISWAP"};
    std::vector<std::string> gate_set3 = {"PS", "RX", "RY", "RZ"};
    std::vector<std::string> gate_set4 = {"Rxx", "Ryy", "Rzz"};

    while (true) {
        if (cmd_idx >= args.size()) {
            if (state == State::W_OBJ) {
                throw std::runtime_error(fmt::format("gate {} require qubit id.", gate.gate));
            }
            if (state == State::W_ANG) {
                throw std::runtime_error(fmt::format("gate {} require rotate angle.", gate.gate));
            }
            if (gate.Valid()) {
                circ.push_back(gate.GetGate());
            }
            break;
        }
        auto arg = args[cmd_idx];

        if (state == State::W_GATE) {
            if (gate.Valid()) {
                circ.push_back(gate.GetGate());
            }
            gate.Reset();
            if (std::find(gate_set1.begin(), gate_set1.end(), arg) != gate_set1.end()) {
                nlohmann::json j = arg;
                gate.gate = j.get<GateID>();
                state = State::W_OBJ;
                n_obj = 1;
                cmd_idx += 1;
                continue;
            }
            if (std::find(gate_set2.begin(), gate_set2.end(), arg) != gate_set2.end()) {
                nlohmann::json j = arg;
                gate.gate = j.get<GateID>();
                state = State::W_OBJ;
                n_obj = 2;
                cmd_idx += 1;
                continue;
            }
            if (std::find(gate_set3.begin(), gate_set3.end(), arg) != gate_set3.end()) {
                nlohmann::json j = arg;
                gate.gate = j.get<GateID>();
                state = State::W_ANG;
                n_obj = 1;
                cmd_idx += 1;
                continue;
            }
            if (std::find(gate_set4.begin(), gate_set4.end(), arg) != gate_set4.end()) {
                nlohmann::json j = arg;
                gate.gate = j.get<GateID>();
                state = State::W_ANG;
                n_obj = 2;
                cmd_idx += 1;
                continue;
            }
            throw std::runtime_error("Cannot convert '" + arg + "' to quantum gate.");
        }
        if (state == State::W_ANG) {
            double ang = std::get<1>(convert_double(arg));
            gate.ang = ang;
            state = State::W_OBJ;
            cmd_idx += 1;
            continue;
        }
        if (state == State::W_OBJ) {
            int obj = std::get<1>(convert_int(arg, n_qubits - 1));
            gate.AddObj(obj);
            n_obj -= 1;
            cmd_idx += 1;
            if (n_obj == 0) {
                state = State::W_CTRL;
            }
            continue;
        }

        if (state == State::W_CTRL) {
            auto [succeed, ctrl] = convert_int(arg, n_qubits - 1, false);
            if (!succeed) {
                if (ctrl >= n_qubits) {
                    throw std::runtime_error("Ctrl qubit larger than system qubits.");
                }
                state = State::W_GATE;
                continue;
            }
            cmd_idx += 1;
            gate.AddCtrl(ctrl);
            continue;
        }
    }
    auto sim = vector::detail::VectorState<vector::detail::CPUVectorPolicyAvxDouble>(n_qubits, seed);
    sim.ApplyCircuit(circ);
    sim.Display();
    return 0;
}
}  // namespace mindquantum::sim::rt
