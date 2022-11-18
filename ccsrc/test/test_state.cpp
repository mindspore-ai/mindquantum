#include <complex>
#include <functional>
#include <iostream>

#include "gate/basic_gate.h"
#include "gate/gates.h"
#include "simulator/densitymatrix/densitymatrix_state.hpp"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

using namespace mindquantum::sim::densitymatrix::detail;
using namespace mindquantum;

auto GetNoPrGate(const std::string& name, VT<Index> obj_qubit, VT<Index> ctrl_qubits) {
    auto g = GetGateByName<double>(name);
    g.obj_qubits_ = obj_qubit;
    g.ctrl_qubits_ = ctrl_qubits;
    std::shared_ptr<BasicGate<double>> p_g = std::make_shared<BasicGate<double>>(g);
    return p_g;
}

auto GetPrGate(const std::string& name, VT<Index> obj_qubit, VT<Index> ctrl_qubits) {
    auto g = GetGateByName<double>(name);
    g.obj_qubits_ = obj_qubit;
    g.ctrl_qubits_ = ctrl_qubits;
    std::shared_ptr<BasicGate<double>> p_g = std::make_shared<BasicGate<double>>(g);
    return p_g;
}

auto GetMeasure(VT<Index> obj_qubit) {
    auto measure = GetMeasureGate<double>(std::string("q0"));
    measure.obj_qubits_ = {obj_qubit};
    std::shared_ptr<BasicGate<double>> p_measure = std::make_shared<BasicGate<double>>(measure);
    return p_measure;
}

auto test_Gate() {
    auto dm = DensityMatrixState<CPUDensityMatrixPolicyBase>(2);
    dm.ApplyGate(GetNoPrGate("H", {0}, {}));
    dm.ApplyGate(GetNoPrGate("X", {1}, {}));
    // dm.ApplyGate(p_x);
    dm.ApplyMeasure(GetMeasure({0}));
    dm.DisplayQS();
}

auto test_GetExpectationReversibleWithGrad() {
    auto dm = DensityMatrixState<CPUDensityMatrixPolicyBase>(3);
    // dm_base::ApplyX(qs, {1}, {}, 8);
    // dm_base::ApplyX(qs, {2}, {}, 8);
    DensityMatrixState<CPUDensityMatrixPolicyBase>::circuit_t circ{{GetPrGate("RX", {0}, {})}};
    DensityMatrixState<CPUDensityMatrixPolicyBase>::circuit_t herm_circ{{GetPrGate("RX", {0}, {})}};
    circ.push_back(GetPrGate("X", {0}, {}));
    MST<size_t> p_map;
    p_map.insert({"RX", 0});
    MST<double> pr_map;
    pr_map.insert({"a", 3});
    ParameterResolver<double> pr{pr_map, 1};
    // dm.ApplyCircuit(circ);
    // dm_base::DisplayQS(qs,3,8);
    PauliWord pw{0, 'Z'};
    VT<PauliWord> v{pw};
    PauliTerm<double> pt{v, 1};
    Hamiltonian<double> ham{{pt}, 3};

    // dm_base::DisplayQS(ham_matrix,3,8);
    auto res = dm.GetExpectationReversibleWithGrad(ham, circ, herm_circ, pr, p_map);
    dm.ApplyCircuit(circ, pr);
    dm.DisplayQS();

    std::cout << res[0] << std::endl;
}

auto test_GetExpectationNonReversibleWithGrad() {
    auto dm = DensityMatrixState<CPUDensityMatrixPolicyBase>(3);
    // dm_base::ApplyX(qs, {1}, {}, 8);
    // dm_base::ApplyX(qs, {2}, {}, 8);
    DensityMatrixState<CPUDensityMatrixPolicyBase>::circuit_t circ{{GetPrGate("RX", {0}, {})}};
    DensityMatrixState<CPUDensityMatrixPolicyBase>::circuit_t herm_circ{{GetPrGate("RX", {0}, {})}};
    circ.push_back(GetPrGate("X", {0}, {}));
    MST<size_t> p_map;
    p_map.insert({"RX", 0});
    MST<double> pr_map;
    pr_map.insert({"a", 3});
    ParameterResolver<double> pr{pr_map, 1};
    // dm.ApplyCircuit(circ);
    // dm_base::DisplayQS(qs,3,8);
    PauliWord pw{0, 'Z'};
    VT<PauliWord> v{pw};
    PauliTerm<double> pt{v, 1};
    Hamiltonian<double> ham{{pt}, 3};

    // dm_base::DisplayQS(ham_matrix,3,8);
    auto res = dm.GetExpectationNonReversibleWithGrad(ham, circ, herm_circ, pr, p_map);
    dm.ApplyCircuit(circ, pr);
    dm.DisplayQS();

    std::cout << res[0] << std::endl;
}

int main() {
    test_GetExpectationReversibleWithGrad();
}