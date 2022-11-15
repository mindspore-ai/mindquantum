
#include <time.h>

#include <iostream>

// #include "simulator/densitymatrix/densitymatrix_state.hpp"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

using namespace mindquantum::sim::densitymatrix::detail;

auto test_Gate() {
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

auto test_Time() {
    auto qs = CPUDensityMatrixPolicyBase::InitState(1024);
    auto start = clock();
    for (int i = 0; i < 1; i++) {
        CPUDensityMatrixPolicyBase::IsPure(qs, 1024);
    }
    auto end = clock();
    auto time = end - start;
    std::cout << time << std::endl;
}

auto test_ExpectDiffSingleQubitMatrix() {
    auto qs = CPUDensityMatrixPolicyBase::InitState(8, 1);
    CPUDensityMatrixPolicyBase::ApplyRX(qs, {0}, {}, 3,8);
    auto ham = CPUDensityMatrixPolicyBase::SelfHermitanHam({{0, 1}, {1, 0}}, {0}, {}, 8);
    auto res = CPUDensityMatrixPolicyBase::ExpectDiffRX(qs, ham, {0}, {}, 3, 8);
    return res;
}

int main() {
    auto res = test_ExpectDiffSingleQubitMatrix();
    std::cout << res << std::endl;

}