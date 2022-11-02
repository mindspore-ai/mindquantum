
#include <iostream>

// #include "simulator/densitymatrix/densitymatrix_state.hpp"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

using namespace mindquantum::sim::densitymatrix::detail;

int main() {

    auto qs = CPUDensityMatrixPolicyBase::InitState(10, 1);
    // CPUDensityMatrixPolicyBase::ApplyX(qs, {1}, {}, 2);
    CPUDensityMatrixPolicyBase::ApplyH(qs, {1}, {}, 2);
    // CPUDensityMatrixPolicyBase::ApplyT(qs, {1}, {}, 2);
    CPUDensityMatrixPolicyBase::DisplayQS(qs, 2, 4);
    std::cout << CPUDensityMatrixPolicyBase::IsPure(qs, 4, 10) << std::endl;
}