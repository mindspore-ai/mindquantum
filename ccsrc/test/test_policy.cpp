
#include <iostream>

// #include "simulator/densitymatrix/densitymatrix_state.hpp"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

using namespace mindquantum::sim::densitymatrix::detail;

int main() {

    auto qs = CPUDensityMatrixPolicyBase::InitState(4, 1);
    CPUDensityMatrixPolicyBase::ApplyH(qs, {0}, {}, 4);
    // CPUDensityMatrixPolicyBase::ApplyRX(qs, {0}, {1}, M_PI, 8);
    CPUDensityMatrixPolicyBase::ApplyY(qs, {1}, {0}, 4);
    CPUDensityMatrixPolicyBase::DisplayQS(qs, 2, 4);
    std::cout << CPUDensityMatrixPolicyBase::IsPure(qs, 4) << std::endl;

    auto v = CPUDensityMatrixPolicyBase::PureStateVector(qs, 4);
    CPUDensityMatrixPolicyBase::Display(v, 2);
}