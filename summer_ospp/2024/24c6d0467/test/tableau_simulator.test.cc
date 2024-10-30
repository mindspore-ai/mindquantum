#include "circuit/operate_unit.h"
#include "circuit/circuit.h"
#include "circuit/gate_type.h"
#include "tableau/tableau_simulator.hpp"
#include <gtest/gtest.h>
#include <random>

TEST(tableau_test, inverse_test) {
    Tableau tableau(4);
    auto xt = tableau.get_xs().get_x_table();
    xt[0][1] = true;
    xt[0][2] = true;
    tableau.to_str();
    tableau.do_transposed();
    tableau.to_str();
}

TEST(simulator_test, x_test) {
    std::mt19937_64 rng;
    std::random_device rd;
    OperateUnit op(GateType::X, 0);
    rng.seed(1);
    TableauSimulator sim(1, rng);
    sim.do_X(op);
    // sim.print_measure_result();
    sim.get_tableau().to_str();
}

TEST(simulaotor_test, y_test) {
    std::mt19937_64 rng;
    std::random_device rd;
    OperateUnit op(GateType::Y, 0);
    OperateUnit op3(GateType::MEASURE, 0);
    rng.seed(1);
    TableauSimulator sim(1, rng);
    sim.do_Y(op);
    sim.get_tableau().to_str();
}

TEST(simulator_test, z_test) {
    std::mt19937_64 rng;
    std::random_device rd;
    OperateUnit op(GateType::Z, 0);
    rng.seed(1);
    TableauSimulator sim(1, rng);
    sim.do_Z(op);
    sim.get_tableau().to_str();
}

TEST(simulaotr_test, h_test) {
    std::mt19937_64 rng;
    std::random_device rd;
    OperateUnit op(GateType::HADAMARD, 0);
    TableauSimulator sim(1, rng);
    sim.do_H(op);
    sim.get_tableau().to_str();
}

TEST(simulator_test, cnot_test) {
    std::mt19937_64 rng;
    std::random_device rd;
    OperateUnit op(GateType::CNOT, 0, 1);
    TableauSimulator sim(2, rng);
    sim.do_CNOT(op);
    sim.get_tableau().to_str();
}

TEST(simulaotr_test, bell_test) {
    std::mt19937_64 rng;
    std::random_device rd;
    OperateUnit op1(GateType::HADAMARD, 0);
    OperateUnit op2(GateType::CNOT, 0, 1);
    OperateUnit op3(GateType::MEASURE, 0);
    rng.seed(1);
    TableauSimulator sim(1, rng);
    sim.do_H(op1);
    sim.do_CNOT(op2);
    sim.do_MEASURE(op3);
    sim.get_tableau().to_str();
}

TEST(tableau_test, ghz_test) {
    std::mt19937_64 rng;
    rng.seed(1234);
    TableauSimulator sim(3, rng);
    OperateUnit op1(GateType::HADAMARD, 0);
    OperateUnit op2(GateType::CNOT, 0, 1);
    OperateUnit op3(GateType::CNOT, 1, 2);
    OperateUnit op4(GateType::MEASURE, 0);
    OperateUnit op5(GateType::MEASURE, 1);
    Circuit circuit{};
    circuit.add_operation(op1);
    circuit.add_operation(op2);
    circuit.add_operation(op3);
    circuit.add_operation(op4);
    circuit.add_operation(op5);
    sim.do_circuit(circuit);
    sim.get_tableau().to_str();
    sim.print_measure_result();
}


TEST(simulator_test, batch_sample_test) {
    std::mt19937_64 rng;
    OperateUnit op1(GateType::HADAMARD, 0);
    OperateUnit op2(GateType::MEASURE, 0);
    for (size_t i = 0; i < 10; ++i) {
        rng.seed(i);
        TableauSimulator sim(1, rng);
        sim.do_H(op1);
        sim.do_MEASURE(op2);
        sim.get_tableau().to_str();
        sim.print_measure_result();
    }
}

TEST(unit_test, test_random) {
    std::mt19937_64 rng(0);
    for (int i = 0; i < 10; ++i) {
        std::cout << ( rng() & 1 );
    } 
    std::cout << std::endl;
}