
#include <time.h>

#include <iostream>
#include <vector>

// #include "simulator/densitymatrix/densitymatrix_state.hpp"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

using namespace mindquantum::sim::densitymatrix::detail;
using dm_base = mindquantum::sim::densitymatrix::detail::CPUDensityMatrixPolicyBase;
auto test_Gate() {
    auto qs = dm_base::InitState(8, 1);
    dm_base::ApplyH(qs, {1}, {}, 8);
    dm_base::ApplyX(qs, {0}, {}, 8);
    // dm_base::ApplyRX(qs, {0}, {1}, M_PI, 8);
    dm_base::ApplySGate(qs, {1}, {0}, 8);
    dm_base::DisplayQS(qs, 3, 8);
    std::cout << dm_base::IsPure(qs, 8) << std::endl;

    auto v = dm_base::PureStateVector(qs, 8);
    dm_base::Display(v, 3);
}

auto test_Time() {
    auto qs = dm_base::InitState(8192);
    auto start = clock();
    for (int i = 0; i < 10; i++) {
        dm_base::ApplyH(qs, {0}, {}, 8192);
    }
    auto end = clock();
    auto time = end - start;
    std::cout << time << std::endl;
}

auto test_SelfAdjointHam() {
    using namespace mindquantum;
    // auto qs = dm_base::InitState(4, 1);
    // dm_base::ApplyRX(qs, {0}, {}, 3, 4);
    PauliWord pw{0, 'Y'};
    PauliWord pw2{1, 'X'};
    VT<PauliWord> v{pw, pw2};
    PauliTerm<double> ham{v, 1};
    auto a = dm_base::SelfAdjointHam({ham}, 4);
    dm_base::DisplayQS(a, 2, 4);
    // auto res = dm_base::ExpectDiffRX(qs, ham, {0}, {}, 3, 8);
    // return res;
}

int main() {
    test_SelfAdjointHam();
}