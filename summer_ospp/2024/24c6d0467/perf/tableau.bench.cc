#include "perf/my_benchmark.hpp"
#include <vector>
#include <iostream>
#include "tableau/tableau_simulator.hpp"
#include "circuit/operate_unit.h"

BENCHMARK(tableau_cnot_1000_benchmark, 100000) {
    do_benchmark([]() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        TableauSimulator sim(1, rng);
        OperateUnit op(GateType::CNOT, 0, 1);
        sim.do_CNOT(op);
    }, 100000);
}

BENCHMARK(bell_1000_benchamrk, 100000) {
    do_benchmark([]() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        TableauSimulator sim(2, rng);
        Circuit circuit{};
        circuit.add_operation(OperateUnit{GateType::HADAMARD, 0});
        circuit.add_operation(OperateUnit{GateType::CNOT, 0, 1});
        circuit.add_operation(OperateUnit{GateType::MEASURE, 0});
        circuit.add_operation(OperateUnit{GateType::MEASURE, 1});
        sim.do_circuit(circuit);
    }, 100000);
}

BENCHMARK(ghz_1000_benchmark, 100000) {
    do_benchmark([]() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        TableauSimulator sim(3, rng);
        Circuit circuit{};
        circuit.add_operation(OperateUnit{GateType::HADAMARD, 0});
        circuit.add_operation(OperateUnit{GateType::CNOT, 0, 1});
        circuit.add_operation(OperateUnit{GateType::CNOT, 1 ,2});
        circuit.add_operation(OperateUnit{GateType::MEASURE, 0});
        circuit.add_operation(OperateUnit{GateType::MEASURE, 1});
        circuit.add_operation(OperateUnit{GateType::MEASURE, 2});
        sim.do_circuit(circuit);
    }, 100000);
}

BENCHMARK(x_10000_test, 100000) {
    do_benchmark([]() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        TableauSimulator sim(1, rng);
        Circuit circuit{};
        circuit.add_operation(OperateUnit{GateType::X, 0});
        circuit.add_operation(OperateUnit{GateType::MEASURE, 0});
        sim.do_circuit(circuit);
    }, 100000);
}