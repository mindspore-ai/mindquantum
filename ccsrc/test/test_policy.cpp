
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
    auto qs = dm_base::InitState(4);
    auto start = clock();
    for (int i = 0; i < 1000000; i++) {
        dm_base::ApplyBitFlip(qs, {0}, 0.1, 4);
    }
    auto end = clock();
    auto time = end - start;
    std::cout << time << std::endl;
}

auto test_SelfAdjointHam() {
    using namespace mindquantum;
    PauliWord pw{0, 'Y'};
    PauliWord pw2{1, 'X'};
    VT<PauliWord> v{pw, pw2};
    PauliTerm<double> ham{v, 1};
    auto a = dm_base::HamiltonianMatrix({ham}, 4);
    dm_base::DisplayQS(a, 2, 4);
}

auto test_ExpectDiffSingleQubitMatrix() {
    using namespace mindquantum;
    auto qs = dm_base::InitState(8, 1);
    // dm_base::ApplyX(qs, {1}, {}, 8);
    // dm_base::ApplyX(qs, {2}, {}, 8);
    dm_base::ApplyRX(qs, {0}, {1, 2}, 3, 8);
    dm_base::DisplayQS(qs,3,8);
    PauliWord pw{0, 'Z'};
    VT<PauliWord> v{pw};
    PauliTerm<double> ham{v, 1};
    auto ham_matrix = dm_base::HamiltonianMatrix({ham}, 8);
    dm_base::DisplayQS(ham_matrix,3,8);
    auto res = dm_base::ExpectDiffRX(qs, ham_matrix, {0}, {}, 3, 8);
    return 2* std::real(res);
}
auto test_GetExpectationReversibleWithGrad(){
    using namespace mindquantum;
    auto qs = dm_base::InitState(8, 1);
    // dm_base::ApplyX(qs, {1}, {}, 8);
    // dm_base::ApplyX(qs, {2}, {}, 8);
    dm_base::ApplyRX(qs, {0}, {1, 2}, 3, 8);
    // dm_base::DisplayQS(qs,3,8);
    PauliWord pw{0, 'Z'};
    VT<PauliWord> v{pw};
    PauliTerm<double> ham{v, 1};
    auto ham_matrix = dm_base::HamiltonianMatrix({ham}, 8);
    // dm_base::DisplayQS(ham_matrix,3,8);
    auto res = dm_base::ExpectDiffRX(qs, ham_matrix, {0}, {}, 3, 8);
    return 2* std::real(res);
}

auto test_Channel(){
    auto qs = dm_base::InitState(8, 1);
    dm_base::ApplyX(qs, {0}, {}, 8);
    dm_base::ApplyAmplitudeDamping(qs, {0}, 0.1, 8);
    
    // dm_base::ApplyRX(qs, {0}, {1}, M_PI, 8);
    // dm_base::ApplySGate(qs, {1}, {0}, 8);
    dm_base::DisplayQS(qs, 3, 8);
    std::cout << dm_base::IsPure(qs, 8) << std::endl;

    // auto v = dm_base::PureStateVector(qs, 8);
    // dm_base::Display(v, 3);
}

int main() {
    test_Time();
}