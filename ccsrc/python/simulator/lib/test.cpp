#include <complex>

#include "core/parameter_resolver.hpp"
#include "ops/gates/details/parameter_resolver_coeff_policy.hpp"
#include "ops/gates/fermion_operator.hpp"
#include "ops/gates/term_value.hpp"
#include "ops/gates/terms_operator_base.hpp"

void test() {
    namespace mq = mindquantum;

    using pr_t = mq::ParameterResolver<double>;
    using pr_cmplx_t = mq::ParameterResolver<std::complex<double>>;
    using all_scalar_types_t = std::tuple<double, std::complex<double>, pr_t, pr_cmplx_t>;
    using fo_d = mq::ops::FermionOperator<pr_t>;
    using fo_cd = mq::ops::FermionOperator<pr_cmplx_t>;
    auto a = fo_d({0, mq::ops::TermValue::a});
    auto b = fo_cd({1, mq::ops::TermValue::adg});
    auto c = a + b;
    std::cout << c.to_string() << std::endl;
}
int main() {
    test();
}
