/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef SIMULATOR_VECTOR_GRADIENT_H_
#define SIMULATOR_VECTOR_GRADIENT_H_
#include <algorithm>
#include <complex>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/mq_base_types.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/matrix.h"
#include "simulator/utils.h"

namespace mindquantum {
/*
 * ! A example policy.
 *
 * struct policy{
 *     using ham_t;
 *     using circuit_t;
 *     using gate_t;
 *     using sim_t;
 *     using py_qs_data_t;
 *     using calc_type;
 *     static std::shared_ptr<sim_t> CopySimToSharedPtr(const sim_t* sim);
 *     static void ApplyCircuit(sim_t* sim, const circuit_t& circ, const parameter::ParameterResolver& pr);
 *     static void ApplyHamiltonian(sim_t* sim, const std::shared_ptr<ham_t>& ham);
 *     static py_qs_data_t Vdot(sim_t* psi_l, sim_t* psi_r);
 *     static void ApplyGate(sim_t* sim, const gate_t& g, const parameter::ParameterResolver& pr);
 *     static bool GateRequiresGrad(const gate_t& g);
 *     static std::pair<MST<size_t>, tensor::Matrix> GetJacobi(const gate_t& g);
 *     static tensor::Matrix ExpectDiffGate(sim_t* psi_l, sim_t* psi_r, const gate_t& g, const
 * parameter::ParameterResolver& pr)；
 * };
 */

template <typename policy>
class GradientHelper {
    using ham_t = typename policy::ham_t;
    using circuit_t = typename policy::circuit_t;
    using sim_t = typename policy::sim_t;
    using py_qs_data_t = typename policy::py_qs_data_t;
    using calc_type = typename policy::calc_type;

 public:
    static VT<VVT<py_qs_data_t>> QramHermitianAdjointGradient(const sim_t* sim_, const VVT<py_qs_data_t>& init_states,
                                                              const circuit_t& circ, const circuit_t& herm_circ,
                                                              const VT<std::shared_ptr<ham_t>>& hams,
                                                              const VT<calc_type>& ans_data, const VS& ans_name,
                                                              size_t batch_threads, size_t mea_threads) {
        auto n_prs = init_states.size();
        auto n_params = ans_name.size();
        auto output = InitFAndG(n_prs, hams.size(), n_params + 1);
        auto p_map = GeneratePMap({}, ans_name);
        const auto prepared_adjoint = PrepareAdjointCircuit(sim_, herm_circ);
        const auto selected_batch_threads = SelectGradientBatchThreads(sim_, batch_threads);
        ParallelTask(selected_batch_threads, n_prs, [&](int idx) {
            parameter::ParameterResolver pr = parameter::ParameterResolver();
            pr.SetItems(ans_name, ans_data);
            std::shared_ptr<sim_t> sim = policy::SimilarSim(sim_, init_states[idx]);
            policy::ApplyCircuit(sim.get(), circ, pr);
            auto psi_r = *sim.get();
            output[idx] = LeftSizeGrad(sim.get(), &psi_r, prepared_adjoint, hams, pr, p_map);
        });
        GradRealDouble(&output, n_params);
        return output;
    }

    static VT<VVT<py_qs_data_t>> NonHermitianAdjointGradient(
        const sim_t* psi_l_, const sim_t* psi_r_, const circuit_t& left_circ, const circuit_t& herm_left_circ,
        const circuit_t& right_circ, const circuit_t& herm_right_circ, const VT<std::shared_ptr<ham_t>>& hams,
        const VT<std::shared_ptr<ham_t>>& herm_hams, const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data,
        const VS& enc_name, const VS& ans_name, size_t batch_threads, size_t mea_threads) {
        auto n_prs = enc_data.size();
        auto n_params = enc_name.size() + ans_name.size();
        auto output = InitFAndG(n_prs, hams.size(), n_params + 1);
        auto p_map = GeneratePMap(enc_name, ans_name);
        const auto prepared_left_adjoint = PrepareAdjointCircuit(psi_l_, herm_left_circ);
        const auto prepared_right_adjoint = PrepareAdjointCircuit(psi_r_, herm_right_circ);
        const auto selected_batch_threads = SelectGradientBatchThreads(psi_r_, batch_threads);
        ParallelTask(selected_batch_threads, n_prs, [&](int idx) {
            output[idx] = NonHermitianAdjointSingleBatch(psi_l_, psi_r_, hams, herm_hams, left_circ, right_circ,
                                                         prepared_left_adjoint, prepared_right_adjoint, enc_data,
                                                         ans_data, enc_name, ans_name, p_map, idx);
        });
        return output;
    }

    static VT<VVT<py_qs_data_t>> HermitianAdjointGradient(const sim_t* sim_, const circuit_t& circ,
                                                          const circuit_t& herm_circ,
                                                          const VT<std::shared_ptr<ham_t>>& hams,
                                                          const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data,
                                                          const VS& enc_name, const VS& ans_name, size_t batch_threads,
                                                          size_t mea_threads) {
        auto n_prs = enc_data.size();
        auto n_params = enc_name.size() + ans_name.size();
        auto output = InitFAndG(n_prs, hams.size(), n_params + 1);
        auto p_map = GeneratePMap(enc_name, ans_name);
        const auto prepared_adjoint = PrepareAdjointCircuit(sim_, herm_circ);
        const auto selected_batch_threads = SelectGradientBatchThreads(sim_, batch_threads);
        ParallelTask(selected_batch_threads, n_prs, [&](int idx) {
            parameter::ParameterResolver pr = parameter::ParameterResolver();
            pr.SetItems(enc_name, enc_data[idx]);
            pr.SetItems(ans_name, ans_data);
            std::shared_ptr<sim_t> sim = policy::CopySimToSharedPtr(sim_);
            policy::ApplyCircuit(sim.get(), circ, pr);
            auto psi_r = *sim.get();
            output[idx] = LeftSizeGrad(sim.get(), &psi_r, prepared_adjoint, hams, pr, p_map);
        });
        GradRealDouble(&output, n_params);
        return output;
    }

 private:
    template <typename candidate, typename = void>
    struct HasPreparedAdjointCircuit : std::false_type {};

    template <typename candidate>
    struct HasPreparedAdjointCircuit<candidate, std::void_t<decltype(candidate::PrepareAdjointCircuit(
                                                    std::declval<const sim_t*>(), std::declval<const circuit_t&>()))>>
        : std::true_type {};

    template <typename candidate, typename prepared_t, typename = void>
    struct HasAdjointSweep : std::false_type {};

    template <typename candidate, typename prepared_t>
    struct HasAdjointSweep<candidate, prepared_t,
                           std::void_t<decltype(candidate::RunAdjointSweep(
                               std::declval<sim_t*>(), std::declval<const VT<sim_t*>&>(),
                               std::declval<const prepared_t&>(), std::declval<const parameter::ParameterResolver&>(),
                               std::declval<const MST<size_t>&>(), false, std::declval<VVT<py_qs_data_t>*>()))>>
        : std::true_type {};

    template <typename candidate, typename = void>
    struct HasGradientBatchThreadSelector : std::false_type {};

    template <typename candidate>
    struct HasGradientBatchThreadSelector<candidate, std::void_t<decltype(candidate::SelectGradientBatchThreads(
                                                         std::declval<const sim_t*>(), std::declval<size_t>()))>>
        : std::true_type {};

    static auto PrepareAdjointCircuit(const sim_t* sim, const circuit_t& circuit) {
        if constexpr (HasPreparedAdjointCircuit<policy>::value) {
            return policy::PrepareAdjointCircuit(sim, circuit);
        } else {
            return &circuit;
        }
    }

    static auto SelectGradientBatchThreads(const sim_t* sim, size_t requested_threads) -> size_t {
        if constexpr (HasGradientBatchThreadSelector<policy>::value) {
            return policy::SelectGradientBatchThreads(sim, requested_threads);
        }
        return requested_threads;
    }

    static void GradRealDouble(VT<VVT<py_qs_data_t>>* f_g, size_t n_grad) {
        auto& output = *f_g;
        for (auto& i : output) {
            for (auto& j : i) {
                for (size_t k = 1; k < n_grad + 1; k++) {
                    j[k] += std::conj(j[k]);
                }
            }
        }
    }

    static VT<VVT<py_qs_data_t>> InitFAndG(size_t dim0, size_t dim1, size_t dim2) {
        VT<VVT<py_qs_data_t>> output;
        for (size_t i = 0; i < dim0; i++) {
            output.push_back({});
            for (size_t j = 0; j < dim1; j++) {
                output[i].push_back({});
                for (size_t k = 0; k < dim2; k++) {
                    output[i][j].push_back(0);
                }
            }
        }
        return output;
    }

    static MST<size_t> GeneratePMap(const VS& enc_name, const VS& ans_name) {
        MST<size_t> p_map;
        for (size_t i = 0; i < enc_name.size(); i++) {
            p_map[enc_name[i]] = i;
        }
        for (size_t i = 0; i < ans_name.size(); i++) {
            p_map[ans_name[i]] = i + enc_name.size();
        }
        return p_map;
    }

    template <typename prepared_t>
    static VVT<py_qs_data_t> LeftSizeGrad(sim_t* psi_l, sim_t* psi_r, const prepared_t& prepared_adjoint,
                                          const VT<std::shared_ptr<ham_t>>& hams,
                                          const parameter::ParameterResolver& pr, const MST<size_t>& p_map) {
        int n_hams = hams.size();
        VVT<py_qs_data_t> f_and_g(n_hams, VT<py_qs_data_t>((1 + p_map.size()), 0));
        VT<std::shared_ptr<sim_t>> psi_rs(n_hams);
        for (int i = 0; i < n_hams; i++) {
            if (i == 0) {
                std::shared_ptr<sim_t> psi_r_ptr(psi_r, [](sim_t* obj) {});
                psi_rs[i] = psi_r_ptr;
            } else {
                psi_rs[i] = policy::CopySimToSharedPtr(psi_r);
            }
        }
        for (int i = 0; i < n_hams; i++) {
            policy::ApplyHamiltonian(psi_rs[i].get(), hams[i]);
        }
        if constexpr (HasAdjointSweep<policy, prepared_t>::value) {
            VT<sim_t*> batch_states;
            batch_states.reserve(psi_rs.size());
            for (const auto& state : psi_rs) {
                batch_states.push_back(state.get());
            }
            policy::RunAdjointSweep(psi_l, batch_states, prepared_adjoint, pr, p_map, false, &f_and_g);
        } else {
            for (int i = 0; i < n_hams; i++) {
                f_and_g[i][0] = policy::Vdot(psi_l, psi_rs[i].get());
            }
            const auto& circ_l_herm = *prepared_adjoint;
            for (const auto& g : circ_l_herm) {
                policy::ApplyGate(psi_l, g, pr);
                if (policy::GateRequiresGrad(g)) {
                    std::pair<MST<size_t>, tensor::Matrix> jacobi = policy::GetJacobi(g);
                    if (const auto& [title, jac] = jacobi; title.size() != 0) {
                        for (int j = 0; j < n_hams; j++) {
                            auto intrin_grad = policy::ExpectDiffGate(psi_l, psi_rs[j].get(), g, pr);
                            auto p_grad = tensor::ops::cpu::to_vector<py_qs_data_t>(
                                tensor::ops::MatMul(intrin_grad, jac));
                            for (const auto& [name, idx] : title) {
                                f_and_g[j][1 + p_map.at(name)] += p_grad[0][idx];
                            }
                        }
                    }
                }
                for (int j = 0; j < n_hams; j++) {
                    policy::ApplyGate(psi_rs[j].get(), g, pr);
                }
            }
        }
        return f_and_g;
    }

    template <typename prepared_t>
    static VVT<py_qs_data_t> RightSizeGrad(sim_t* psi_l, sim_t* psi_r, const prepared_t& prepared_adjoint,
                                           const VT<std::shared_ptr<ham_t>>& herm_hams,
                                           const parameter::ParameterResolver& pr, const MST<size_t>& p_map) {
        int n_hams = herm_hams.size();
        VVT<py_qs_data_t> f_and_g(n_hams, VT<py_qs_data_t>((1 + p_map.size()), 0));
        VT<std::shared_ptr<sim_t>> psi_ls(n_hams);
        for (int i = 0; i < n_hams; i++) {
            if (i == 0) {
                std::shared_ptr<sim_t> psi_l_ptr(psi_l, [](sim_t* obj) {});
                psi_ls[i] = psi_l_ptr;
            } else {
                psi_ls[i] = policy::CopySimToSharedPtr(psi_l);
            }
        }
        for (int i = 0; i < n_hams; i++) {
            policy::ApplyHamiltonian(psi_ls[i].get(), herm_hams[i]);
        }
        if constexpr (HasAdjointSweep<policy, prepared_t>::value) {
            VT<sim_t*> batch_states;
            batch_states.reserve(psi_ls.size());
            for (const auto& state : psi_ls) {
                batch_states.push_back(state.get());
            }
            policy::RunAdjointSweep(psi_r, batch_states, prepared_adjoint, pr, p_map, true, &f_and_g);
        } else {
            for (int i = 0; i < n_hams; i++) {
                f_and_g[i][0] = policy::Vdot(psi_ls[i].get(), psi_r);
            }
            const auto& circ_r_herm = *prepared_adjoint;
            for (const auto& g : circ_r_herm) {
                policy::ApplyGate(psi_r, g, pr);
                if (policy::GateRequiresGrad(g)) {
                    std::pair<MST<size_t>, tensor::Matrix> jacobi = policy::GetJacobi(g);
                    if (const auto& [title, jac] = jacobi; title.size() != 0) {
                        for (int j = 0; j < n_hams; j++) {
                            auto intrin_grad = policy::ExpectDiffGate(psi_r, psi_ls[j].get(), g, pr);
                            auto p_grad = tensor::ops::cpu::to_vector<py_qs_data_t>(
                                tensor::ops::MatMul(intrin_grad, jac));
                            for (const auto& [name, idx] : title) {
                                f_and_g[j][1 + p_map.at(name)] += std::conj(p_grad[0][idx]);
                            }
                        }
                    }
                }
                for (int j = 0; j < n_hams; j++) {
                    policy::ApplyGate(psi_ls[j].get(), g, pr);
                }
            }
        }
        return f_and_g;
    }

    template <typename prepared_left_t, typename prepared_right_t>
    static VVT<py_qs_data_t> NonHermitianAdjointSingleBatch(
        const sim_t* psi_l_, const sim_t* psi_r_, const VT<std::shared_ptr<ham_t>>& hams,
        const VT<std::shared_ptr<ham_t>>& herm_hams, const circuit_t& left_circ, const circuit_t& right_circ,
        const prepared_left_t& prepared_left_adjoint, const prepared_right_t& prepared_right_adjoint,
        const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
        const MST<size_t>& p_map, int batch_idx = 0) {
        parameter::ParameterResolver pr = parameter::ParameterResolver();
        pr.SetItems(enc_name, enc_data[batch_idx]);
        pr.SetItems(ans_name, ans_data);
        std::shared_ptr<sim_t> psi_l = policy::CopySimToSharedPtr(psi_l_);
        std::shared_ptr<sim_t> psi_r = policy::CopySimToSharedPtr(psi_r_);
        policy::ApplyCircuit(psi_l.get(), left_circ, pr);
        policy::ApplyCircuit(psi_r.get(), right_circ, pr);
        VVT<py_qs_data_t> f_g1, f_g2;
        {
            std::shared_ptr<sim_t> psi_l1 = policy::CopySimToSharedPtr(psi_l.get());
            std::shared_ptr<sim_t> psi_r1 = policy::CopySimToSharedPtr(psi_r.get());
            f_g1 = LeftSizeGrad(psi_l1.get(), psi_r1.get(), prepared_left_adjoint, hams, pr, p_map);
        }
        {
            std::shared_ptr<sim_t> psi_l2 = policy::CopySimToSharedPtr(psi_l.get());
            std::shared_ptr<sim_t> psi_r2 = policy::CopySimToSharedPtr(psi_r.get());
            f_g2 = RightSizeGrad(psi_l2.get(), psi_r2.get(), prepared_right_adjoint, herm_hams, pr, p_map);
        }
        for (int i = 0; i < hams.size(); i++) {
            for (int j = 1; j < p_map.size() + 1; j++) {
                f_g1[i][j] += f_g2[i][j];
            }
        }
        return f_g1;
    }

    static void ParallelTask(size_t batch_threads, size_t n_prs, const std::function<void(int)>& run) {
        sim::ParallelBatch(batch_threads, n_prs, [&run](size_t index) { run(static_cast<int>(index)); });
    }
};
}  // namespace mindquantum
#endif
