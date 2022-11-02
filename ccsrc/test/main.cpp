
#include <iostream>

// #include "simulator/densitymatrix/densitymatrix_state.hpp"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

using namespace mindquantum::sim::densitymatrix::detail;

int main() {

    auto qs = CPUDensityMatrixPolicyBase::InitState(36, 1);
    CPUDensityMatrixPolicyBase::ApplyH(qs, {1}, {}, 8);
    CPUDensityMatrixPolicyBase::ApplyRX(qs, {0}, {1}, 4, 8);
    // CPUDensityMatrixPolicyBase::ApplyT(qs, {1}, {}, 2);
    CPUDensityMatrixPolicyBase::DisplayQS(qs, 4, 8);
    std::cout << CPUDensityMatrixPolicyBase::IsPure(qs, 8, 36) << std::endl;

    auto v = CPUDensityMatrixPolicyBase::PureStateVector(qs, 8);
    CPUDensityMatrixPolicyBase::Display(v, 4);
}