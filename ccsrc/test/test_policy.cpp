
#include <iostream>

// #include "simulator/densitymatrix/densitymatrix_state.hpp"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

using namespace mindquantum::sim::densitymatrix::detail;

int main() {

    auto qs = CPUDensityMatrixPolicyBase::InitState(8, 1);
    CPUDensityMatrixPolicyBase::ApplyH(qs, {1}, {}, 8);
    CPUDensityMatrixPolicyBase::ApplyX(qs, {0}, {}, 8);
    // CPUDensityMatrixPolicyBase::ApplyRX(qs, {0}, {1}, M_PI, 8);
    CPUDensityMatrixPolicyBase::ApplySGate(qs, {1}, {0}, 8);
    CPUDensityMatrixPolicyBase::DisplayQS(qs, 3, 8);
    std::cout << CPUDensityMatrixPolicyBase::IsPure(qs, 8) << std::endl;

    auto v = CPUDensityMatrixPolicyBase::PureStateVector(qs, 8);
    CPUDensityMatrixPolicyBase::Display(v, 3);
}