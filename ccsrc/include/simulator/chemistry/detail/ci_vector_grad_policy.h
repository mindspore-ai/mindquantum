/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CI_VECTOR_GRAD_POLICY_H
#define INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CI_VECTOR_GRAD_POLICY_H

#include <cmath>

#include <algorithm>
#include <complex>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "math/pr/parameter_resolver.h"
#include "simulator/chemistry/detail/cpp_ci_hamiltonian.h"
#include "simulator/chemistry/detail/cpp_excitation_operator.h"
#include "simulator/chemistry/detail/cpu_ci_vector_double_policy.h"
#include "simulator/chemistry/detail/cpu_ci_vector_float_policy.h"
// Forward declare CIState to avoid circular include
namespace mindquantum::sim::chem {
template <typename policy_t>
class CIState;
}

namespace mindquantum::sim::chem::detail {

/// Policy struct to compute ⟨H⟩ and its gradient w.r.t. UCC parameters.
template <typename calc_t>
struct CIVectorGradPolicy {
    using ham_t = CppCIHamiltonian<calc_t>;
    using circuit_t = std::vector<std::shared_ptr<CppExcitationOperator<calc_t>>>;
    using gate_t = std::shared_ptr<CppExcitationOperator<calc_t>>;
    using sim_t = ::mindquantum::sim::chem::CIState<
        std::conditional_t<std::is_same_v<calc_t, float>, cpu_ci_vector_float_policy, cpu_ci_vector_double_policy>>;
    using py_qs_data_t = std::complex<calc_t>;
    using calc_type = calc_t;
    using CIVector = typename ham_t::CIVector;

    static std::shared_ptr<sim_t> CopySimToSharedPtr(const sim_t* sim) {
        return std::make_shared<sim_t>(*sim);
    }

    static std::shared_ptr<sim_t> SimilarSim(const sim_t* sim, const std::vector<calc_type>& qs_data) {
        return std::make_shared<sim_t>(sim->n_qubits_, sim->n_electrons_, sim->seed_, qs_data);
    }

    // Apply the UCC circuit by resolving gate parameters via ParameterResolver
    static void ApplyCircuit(sim_t* sim, const circuit_t& circ, const parameter::ParameterResolver& pr) {
        for (auto& g : circ) {
            sim->ApplySingleUCCGate(g, pr);
        }
    }

    static void ApplyHamiltonian(sim_t* sim, const std::shared_ptr<ham_t>& ham) {
        // Apply H to CI-vector state in-place
        auto tmp = ham->apply_to_civector(*sim->qs_);
        *sim->qs_ = std::move(tmp);
    }

    static py_qs_data_t Vdot(sim_t* psi_l, sim_t* psi_r) {
        // CI-vector overlap ⟨psi_l|psi_r⟩
        py_qs_data_t val{};
        const auto& data_l = psi_l->qs_->data();
        const auto& data_r = psi_r->qs_->data();
        size_t dim = data_l.size();
        for (size_t i = 0; i < dim; ++i) {
            auto amp_l = data_l[i];
            if (amp_l == calc_type(0)) {
                continue;
            }
            std::complex<calc_t> cl = static_cast<std::complex<calc_t>>(amp_l);
            std::complex<calc_t> cr = static_cast<std::complex<calc_t>>(data_r[i]);
            val += std::conj(cl) * cr;
        }
        return val;
    }

    // Apply a single UCC gate using its pre-set parameter value (theta_value)
    static void ApplyGate(sim_t* sim, const gate_t& g, const parameter::ParameterResolver& pr) {
        sim->ApplySingleUCCGate(g, pr);
    }

    static bool GateRequiresGrad(const gate_t& g) {
        return g->coeff.HasRequireGradParams();
    }

    static std::pair<MST<size_t>, tensor::Matrix> GetJacobi(const gate_t& g) {
        return parameter::Jacobi(std::vector<parameter::ParameterResolver>{g->coeff});
    }

    static tensor::Matrix ExpectDiffGate(sim_t* psi_l_sim, sim_t* psi_r_sim, const gate_t& g,
                                         const parameter::ParameterResolver& pr) {
        // This function calculates <psi_l| G \exp{\theta G} |psi_r> where G is the generator of the UCC gate.
        // The calculation is optimized to a single loop over coupled basis states without intermediate allocations.
        if (!g->is_valid_) {
            // This case should ideally not be hit if all gates are standard UCC excitations.
            // Returning zero for safety.
            return tensor::Matrix(std::vector<std::vector<py_qs_data_t>>{{py_qs_data_t(0)}});
        }

        const double theta = tensor::ops::cpu::to_vector<double>(g->coeff.Combination(pr).const_value)[0];
        const auto cos_th = static_cast<calc_type>(std::cos(theta));
        const auto sin_th = static_cast<calc_type>(std::sin(theta));

        const auto& data_l = psi_l_sim->qs_->data();
        const auto& data_r = psi_r_sim->qs_->data();
        auto indexer = ci_basis::IndexingManager::GetIndexer(g->n_qubits_, g->n_electrons_);
        const size_t dim = psi_l_sim->qs_->dimension();

        calc_type real_part = 0;
        calc_type imag_part = 0;

        // The formula for each group is derived from <d|G|c'> where |c'> = exp(theta*G)|c>.
        // For each group {s1, s2}, the contribution is:
        // phase * (conj(d2)*c'_1 - conj(d1)*c'_2)
        // with c'_1 = cos_th*c1 - phase*sin_th*c2
        // and  c'_2 = phase*sin_th*c1 + cos_th*c2
        // This can be parallelized with reduction.
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:real_part, imag_part) schedule(static)), dim, 1UL << 13,
                for (omp::idx_t i = 0; i < dim; i++) {
                    uint64_t mask_i = indexer->unrank(i);
                    if ((mask_i & g->excit_mask_) == g->ket_bra_mask_) {
                        uint64_t mask_j = mask_i ^ g->flip_mask_;
                        size_t j = indexer->rank(mask_j);

                        const auto c1 = data_r[i], c2 = data_r[j];
                        const auto d1 = data_l[i], d2 = data_l[j];
                        double phase_d = 1.0;
                        uint64_t current_mask = mask_i;
                        for (size_t k = 0; k < g->op_sequence_.size(); ++k) {
                            const auto& [orb_idx, is_creation] = g->op_sequence_[k];
                            const auto& parity_mask = g->op_parity_masks_[k];
                            if (ci_basis::SlaterDeterminant::count_set_bits(current_mask & parity_mask) & 1) {
                                phase_d = -phase_d;
                            }
                            current_mask ^= (1ULL << orb_idx);
                        }
                        const auto phase = static_cast<calc_type>(phase_d);

                        const auto c_prime_1 = cos_th * c1 - phase * sin_th * c2;
                        const auto c_prime_2 = phase * sin_th * c1 + cos_th * c2;
                        const auto term = phase * (std::conj(d2) * c_prime_1 - std::conj(d1) * c_prime_2);
                        real_part += term.real();
                        imag_part += term.imag();
                    }
                });

        py_qs_data_t total_val(real_part, imag_part);
        std::vector<std::vector<py_qs_data_t>> mdata{{total_val}};
        return tensor::Matrix(mdata);
    }
};

}  // namespace mindquantum::sim::chem::detail
#endif  // INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CI_VECTOR_GRAD_POLICY_H
