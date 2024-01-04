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
#include <complex>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "core/mq_base_types.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/matrix.h"

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
        ParallelTask(batch_threads, n_prs, [&](int idx) {
            parameter::ParameterResolver pr = parameter::ParameterResolver();
            pr.SetItems(ans_name, ans_data);
            std::shared_ptr<sim_t> sim = policy::SimilarSim(sim_, init_states[idx]);
            policy::ApplyCircuit(sim.get(), circ, pr);
            auto psi_r = *sim.get();
            output[idx] = LeftSizeGrad(sim.get(), &psi_r, herm_circ, hams, pr, p_map);
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
        ParallelTask(batch_threads, n_prs, [&](int idx) {
            output[idx] = NonHermitianAdjointSingleBatch(psi_l_, psi_r_, hams, herm_hams, left_circ, herm_left_circ,
                                                         right_circ, herm_right_circ, enc_data, ans_data, enc_name,
                                                         ans_name, p_map, idx);
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
        ParallelTask(batch_threads, n_prs, [&](int idx) {
            parameter::ParameterResolver pr = parameter::ParameterResolver();
            pr.SetItems(enc_name, enc_data[idx]);
            pr.SetItems(ans_name, ans_data);
            std::shared_ptr<sim_t> sim = policy::CopySimToSharedPtr(sim_);
            policy::ApplyCircuit(sim.get(), circ, pr);
            auto psi_r = *sim.get();
            output[idx] = LeftSizeGrad(sim.get(), &psi_r, herm_circ, hams, pr, p_map);
        });
        GradRealDouble(&output, n_params);
        return output;
    }

 private:
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

    static VVT<py_qs_data_t> LeftSizeGrad(sim_t* psi_l, sim_t* psi_r, const circuit_t& circ_l_herm,
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
            f_and_g[i][0] = policy::Vdot(psi_l, psi_rs[i].get());
        }
        for (const auto& g : circ_l_herm) {
            policy::ApplyGate(psi_l, g, pr);
            if (policy::GateRequiresGrad(g)) {
                std::pair<MST<size_t>, tensor::Matrix> jacobi = policy::GetJacobi(g);
                if (const auto& [title, jac] = jacobi; title.size() != 0) {
                    for (int j = 0; j < n_hams; j++) {
                        auto intrin_grad = policy::ExpectDiffGate(psi_l, psi_rs[j].get(), g, pr);
                        auto p_grad = tensor::ops::cpu::to_vector<py_qs_data_t>(tensor::ops::MatMul(intrin_grad, jac));
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
        return f_and_g;
    }

    static VVT<py_qs_data_t> RightSizeGrad(sim_t* psi_l, sim_t* psi_r, const circuit_t& circ_r_herm,
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
            f_and_g[i][0] = policy::Vdot(psi_ls[i].get(), psi_r);
        }
        for (const auto& g : circ_r_herm) {
            policy::ApplyGate(psi_r, g, pr);
            if (policy::GateRequiresGrad(g)) {
                std::pair<MST<size_t>, tensor::Matrix> jacobi = policy::GetJacobi(g);
                if (const auto& [title, jac] = jacobi; title.size() != 0) {
                    for (int j = 0; j < n_hams; j++) {
                        auto intrin_grad = policy::ExpectDiffGate(psi_r, psi_ls[j].get(), g, pr);
                        auto p_grad = tensor::ops::cpu::to_vector<py_qs_data_t>(tensor::ops::MatMul(intrin_grad, jac));
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
        return f_and_g;
    }

    static VVT<py_qs_data_t> NonHermitianAdjointSingleBatch(
        const sim_t* psi_l_, const sim_t* psi_r_, const VT<std::shared_ptr<ham_t>>& hams,
        const VT<std::shared_ptr<ham_t>>& herm_hams, const circuit_t& left_circ, const circuit_t& herm_left_circ,
        const circuit_t& right_circ, const circuit_t& herm_right_circ, const VVT<calc_type>& enc_data,
        const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name, const MST<size_t>& p_map,
        int batch_idx = 0) {
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
            f_g1 = LeftSizeGrad(psi_l1.get(), psi_r1.get(), herm_left_circ, hams, pr, p_map);
        }
        {
            std::shared_ptr<sim_t> psi_l2 = policy::CopySimToSharedPtr(psi_l.get());
            std::shared_ptr<sim_t> psi_r2 = policy::CopySimToSharedPtr(psi_r.get());
            f_g2 = RightSizeGrad(psi_l2.get(), psi_r2.get(), herm_right_circ, herm_hams, pr, p_map);
        }
        std::cout << f_g1[0][1] << std::endl;
        std::cout << f_g2[0][1] << std::endl;
        for (int i = 0; i < hams.size(); i++) {
            for (int j = 1; j < p_map.size() + 1; j++) {
                f_g1[i][j] += f_g2[i][j];
            }
        }
        return f_g1;
    }

    static void ParallelTask(size_t batch_threads, size_t n_prs, std::function<void(int)> run) {
        if (n_prs == 1) {
            run(0);
            return;
        }
        if (batch_threads == 0) {
            throw std::runtime_error("batch_threads cannot be zero.");
        }
        std::vector<std::thread> tasks;
        tasks.reserve(batch_threads);
        size_t end = 0;
        size_t offset = n_prs / batch_threads;
        size_t left = n_prs % batch_threads;
        for (size_t i = 0; i < batch_threads; ++i) {
            size_t start = end;
            end = start + offset;
            if (i < left) {
                end += 1;
            }
            auto task = [&, start, end]() {
                for (size_t n = start; n < end; n++) {
                    run(n);
                }
            };
            tasks.emplace_back(task);
        }
        for (auto& t : tasks) {
            t.join();
        }
    }
};
}  // namespace mindquantum
#endif
