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

#include "cengines/cpp_graph_mapper.hpp"
#include "cengines/cpp_printer.hpp"
#include "cppcore.hpp"

int main(int argc, char *argv[]) {
    namespace td = tweedledum;

    mindquantum::cengines::CppGraphMapper mapper(5, {{0, 1}, {0, 2}, {0, 3}, {0, 4}}, td::sabre_config{});

    mindquantum::cengines::CppPrinter printer("projectq");

    mindquantum::CppCore::engine_list_t engine_list = {mapper, printer};

    mindquantum::CppCore core;
    core.set_engine_list(engine_list);

    core.allocate_qubit(0);
    core.allocate_qubit(1);
    core.allocate_qubit(2);
    core.allocate_qubit(3);
    core.allocate_qubit(4);

    mindquantum::ops::Command cmd;

    cmd.set_gate(mindquantum::gate_lib::x);
    cmd.set_control_qubits({0});
    cmd.set_qubits({1});
    core.apply_command(cmd);

    cmd.set_gate(mindquantum::gate_lib::x);
    cmd.set_control_qubits({0});
    cmd.set_qubits({2});
    core.apply_command(cmd);

    cmd.set_gate(mindquantum::gate_lib::x);
    cmd.set_control_qubits({0});
    cmd.set_qubits({3});
    core.apply_command(cmd);

    cmd.set_gate(mindquantum::gate_lib::x);
    cmd.set_control_qubits({0});
    cmd.set_qubits({4});
    core.apply_command(cmd);

    cmd.set_gate(mindquantum::gate_lib::x);
    cmd.set_control_qubits({3});
    cmd.set_qubits({4});
    core.apply_command(cmd);

    cmd.set_gate(mindquantum::gate_lib::x);
    cmd.set_control_qubits({1});
    cmd.set_qubits({2});
    core.apply_command(cmd);

    core.flush();
    core.flush();

    return 0;
}
