/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDQUANTUM_BACKENDS_PROJECTQ_PROJECTQ_H_
#define MINDQUANTUM_BACKENDS_PROJECTQ_PROJECTQ_H_

#include <cmath>

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "config/openmp.hpp"

#include "Threadpool.h"
#include "core/parameter_resolver.hpp"
#include "core/utils.hpp"
#include "ops/basic_gate.hpp"
#include "ops/gates.hpp"
#include "ops/hamiltonian.hpp"
#include "projectq/backends/_sim/_cppkernels/simulator.hpp"
#include "projectq_utils.h"

namespace mindquantum {
namespace projectq {
template <typename T>
class Projectq : public ::projectq::Simulator {
 private:
    unsigned seed;
    unsigned n_qubits_;
    VT<unsigned> ordering_;
    unsigned len_;
    RndEngine rnd_eng_;
    std::function<double()> rng_;

 public:
    Projectq() : Simulator(1, 1), n_qubits_(1), rnd_eng_(1), seed(1) {
        for (unsigned i = 0; i < n_qubits_; i++) {
            ordering_.push_back(i);
        }
        len_ = (1UL << (n_qubits_ + 1));
        std::uniform_real_distribution<double> dist(0., 1.);
        rng_ = std::bind(dist, std::ref(rnd_eng_));
    }

    Projectq(unsigned seed, unsigned N) : Simulator(seed, N), n_qubits_(N), rnd_eng_(seed), seed(seed) {
        for (unsigned i = 0; i < n_qubits_; i++) {
            ordering_.push_back(i);
        }
        len_ = (1UL << (n_qubits_ + 1));
        std::uniform_real_distribution<double> dist(0., 1.);
        rng_ = std::bind(dist, std::ref(rnd_eng_));
    }
    Projectq(unsigned seed, unsigned N, calc_type *vec) : Simulator(seed, N), n_qubits_(N), rnd_eng_(seed), seed(seed) {
        for (unsigned i = 0; i < n_qubits_; i++) {
            ordering_.push_back(i);
        }
        len_ = (1UL << (n_qubits_ + 1));
        set_wavefunction(vec, ordering_);
        std::uniform_real_distribution<double> dist(0., 1.);
        rng_ = std::bind(dist, std::ref(rnd_eng_));
    }
    void InitializeSimulator() {
        if (vec_ != NULL) {
            free(vec_);
        }
        vec_ = (StateVector) calloc(len_, sizeof(calc_type));
        vec_[0] = 1;
    }

    void InitializeSimulator(const VT<BasicGate<T>> &circ) {
        Projectq::InitializeSimulator();
        Projectq::ApplyCircuit(circ);
    }

    void InitializeSimulator(CTP<T> vec) {
    }

    void SetState(VT<CT<T>> vec) {
        set_wavefunction(reinterpret_cast<calc_type *>(vec.data()), ordering_);
    }

    void SetThreadsNumber(size_t number) {
        Threadpool::GetInstance().ReSize(number);
    }

    void ApplyGate(const BasicGate<T> &gate) {
        if (gate.is_channel_) {
            Projectq::ApplyChannel(gate);
        } else {
            Projectq::apply_controlled_gate(MCast<T>(gate.base_matrix_.matrix_), VCast(gate.obj_qubits_),
                                            VCast(gate.ctrl_qubits_));
        }
    }

    void ApplyGate(const BasicGate<T> &gate, const ParameterResolver<T> &pr, bool diff = false) {
        T theta = gate.params_.Combination(pr).const_value;
        if (diff) {
            Projectq::apply_controlled_gate(MCast<T>(gate.param_diff_matrix_(theta).matrix_), VCast(gate.obj_qubits_),
                                            VCast(gate.ctrl_qubits_));
            if (gate.ctrl_qubits_.size() != 0) {
                auto ctrl_mask = GetControlMask(gate.ctrl_qubits_);
                THRESHOLD_OMP_FOR(
                    n_qubits_, nQubitTh, for (Index i = 0; i < (len_ >> 1); i++) {
                        if ((i & ctrl_mask) != ctrl_mask) {
                            this->vec_[2 * i] = 0;
                            this->vec_[2 * i + 1] = 0;
                        }
                    })
            }
        } else {
            Projectq::apply_controlled_gate(MCast<T>(gate.param_matrix_(theta).matrix_), VCast(gate.obj_qubits_),
                                            VCast(gate.ctrl_qubits_));
        }
    }

    void ApplyChannel(const BasicGate<T> &gate) {
        if (gate.name_ == "PL") {  // pauli channel; gate is constructed be like: BasicGate(cPL, true, px, py, pz)
            Projectq::ApplyPauliChannel(gate);
        } else if (gate.name_ == "ADC" || gate.name_ == "PDC") {  // damping channel
            Projectq::ApplyDampingChannel(gate);
        } else if (gate.kraus_operator_set_.size() != 0) {  // Kraus channel
            Projectq::ApplyKrausChannel(gate);
        } else {
            throw std::runtime_error("ApplyChannel: Unknown noise channel!");
        }
    }

    void ApplyPauliChannel(const BasicGate<T> &gate) {
        VT<BasicGate<T>> gate_list_ = {XGate<T>, YGate<T>, ZGate<T>, IGate<T>};
        double r = static_cast<double>(rng_());
        //            std::cout << "r = " << r << std::endl;
        auto it = std::lower_bound(gate.cumulative_probs_.begin(), gate.cumulative_probs_.end(), r);
        size_t gate_index;
        if (it != gate.cumulative_probs_.begin()) {
            gate_index = std::distance(gate.cumulative_probs_.begin(), it) - 1;
        } else {
            gate_index = 0;
        }
        BasicGate<T> gate_ = gate_list_[gate_index];  // Select the gate to execute according to r.
                                                      //            std::cout << gate_.name_ << std::endl;
        Projectq::apply_controlled_gate(MCast<T>(gate_.base_matrix_.matrix_), VCast(gate.obj_qubits_),
                                        VCast(gate.ctrl_qubits_));
    }

    void ApplyDampingChannel(const BasicGate<T> &gate) {
        VT<size_t> idx_of_zero;  // the 01 base index of object qubit that being '0'
        VT<size_t> idx_of_one;   // the 01 base index of object qubit that being '1'
        for (size_t i = 0; i < (1 << n_qubits_); ++i) {
            if ((i >> gate.obj_qubits_[0]) & 1) {
                idx_of_one.push_back(i);
            } else {
                idx_of_zero.push_back(i);
            }
        }
        VT<CT<T>> curr_state = Projectq::cheat();
        double reduced_factor_b_square = 0;  // reduced quantum state (a, b) of the object qubit
        for (size_t idx : idx_of_one) {
            reduced_factor_b_square += (curr_state[idx] * std::conj(curr_state[idx])).real();
        }
        double reduced_factor_b = sqrt(reduced_factor_b_square);
        if (reduced_factor_b < 1e-8) {
            return;
        }
        double prob = gate.damping_coeff_ * reduced_factor_b_square;
        double r = static_cast<double>(rng_());
        if (r <= prob) {  // change to '0' or '1' state
            VT<CT<T>> updated_state(curr_state.size());
            if (gate.name_ == "ADC") {  // Amplitude damping channel case
                for (size_t j = 0; j < idx_of_zero.size(); ++j) {
                    updated_state[idx_of_zero[j]] = curr_state[idx_of_one[j]] / reduced_factor_b;
                }
                for (size_t idx : idx_of_one) {
                    updated_state[idx] = 0;
                }
            } else {  // Phase damping channel case
                for (size_t idx : idx_of_one) {
                    updated_state[idx] = curr_state[idx] / reduced_factor_b;
                }
                for (size_t idx : idx_of_zero) {
                    updated_state[idx] = 0;
                }
            }
            Projectq::SetState(updated_state);
        } else {  // change to remained state
            double coeff_a = 1 / sqrt(1 - prob);
            double coeff_b = sqrt(1 - gate.damping_coeff_) / sqrt(1 - prob);
            size_t counter = 0;
            for (size_t i = 0; i < (curr_state.size()); ++i) {
                if (i == idx_of_zero[counter]) {
                    curr_state[i] *= coeff_a;
                    counter += 1;
                } else {
                    curr_state[i] *= coeff_b;
                }
            }
            Projectq::SetState(curr_state);
        }
    }

    void ApplyKrausChannel(const BasicGate<T> &gate) {
        VT<size_t> idx_of_zero;  // the 01 base index of object qubit that being '0'
        VT<size_t> idx_of_one;   // the 01 base index of object qubit that being '1'
        for (size_t i = 0; i < (1 << n_qubits_); ++i) {
            if ((i >> gate.obj_qubits_[0]) & 1) {
                idx_of_one.push_back(i);
            } else {
                idx_of_zero.push_back(i);
            }
        }
        VT<CT<T>> curr_state = Projectq::cheat();
        unsigned n_kraus = (gate.kraus_operator_set_).size();
        double prob = 0;
        for (unsigned i = 0; i < n_kraus; ++i) {
            VT<CT<T>> updated_state(curr_state.size());
            const VVT<CT<T>> &kraus_mat = gate.kraus_operator_set_[i];
            double renormal_factor_square = 0;
            for (size_t j = 0; j < idx_of_zero.size(); ++j) {
                updated_state[idx_of_zero[j]] = curr_state[idx_of_zero[j]] * kraus_mat[0][0]
                                                + curr_state[idx_of_one[j]] * kraus_mat[0][1];
                renormal_factor_square
                    += (updated_state[idx_of_zero[j]] * std::conj(updated_state[idx_of_zero[j]])).real();
            }
            for (size_t j = 0; j < idx_of_one.size(); ++j) {
                updated_state[idx_of_one[j]] = curr_state[idx_of_zero[j]] * kraus_mat[1][0]
                                               + curr_state[idx_of_one[j]] * kraus_mat[1][1];
                renormal_factor_square
                    += (updated_state[idx_of_one[j]] * std::conj(updated_state[idx_of_one[j]])).real();
            }
            double renormal_factor = sqrt(renormal_factor_square);
            prob = renormal_factor_square / (1 - prob);
            double r = static_cast<double>(rng_());
            if (r <= prob) {
                for (auto it = updated_state.begin(); it != updated_state.end(); ++it) {
                    *it /= renormal_factor;
                }
                Projectq::SetState(updated_state);
                break;
            }
        }
    }

    unsigned ApplyMeasure(const BasicGate<T> &gate) {
        run();
        auto qubit = gate.obj_qubits_[0];
        auto mask = (1UL << qubit);
        calc_type zero_amps = 0;
        // #pragma omp parallel for schedule(static) reduction(+ : zero_amps)
        for (unsigned i = 0; i < (len_ >> 1); i++) {
            if ((i & mask) == 0) {
                zero_amps += vec_[2 * i] * vec_[2 * i] + vec_[2 * i + 1] * vec_[2 * i + 1];
            }
        }
        unsigned collapse = (static_cast<unsigned>(rng_() > zero_amps) << qubit);
        auto norm = (collapse == 0) ? sqrt(zero_amps) : sqrt(1 - zero_amps);
        THRESHOLD_OMP_FOR(
            n_qubits_, nQubitTh, for (omp::idx_t i = 0; i < (len_ >> 1); i++) {
                if ((i & mask) == collapse) {
                    vec_[2 * i] /= norm;
                    vec_[2 * i + 1] /= norm;
                } else {
                    vec_[2 * i] = 0;
                    vec_[2 * i + 1] = 0;
                }
            })
        return (collapse >> qubit);
    }

    void ApplyCircuit(const VT<BasicGate<T>> &circ) {
        for (auto &gate : circ) {
            Projectq::ApplyGate(gate);
        }
        Projectq::run();
    }

    void ApplyCircuit(const VT<BasicGate<T>> &circ, const ParameterResolver<T> &pr) {
        for (auto &gate : circ) {
            if (gate.parameterized_) {
                Projectq::ApplyGate(gate, pr);
            } else {
                Projectq::ApplyGate(gate);
            }
        }
        Projectq::run();
    }

    VT<unsigned> Sampling(const VT<BasicGate<T>> &circ, const ParameterResolver<T> &pr, size_t shots,
                          const MST<size_t> &key_map, unsigned seed) {
        auto key_size = key_map.size();
        VT<unsigned> res(shots * key_size);
        RndEngine rnd_eng = RndEngine(seed);
        std::uniform_real_distribution<double> dist(1.0, (1 << 20) * 1.0);
        std::function<double()> rng = std::bind(dist, std::ref(rnd_eng));
        for (size_t i = 0; i < shots; i++) {
            Projectq<T> sim = Projectq<T>(static_cast<unsigned>(rng()), n_qubits_, vec_);
            auto res0 = sim.ApplyCircuitWithMeasure(circ, pr, key_map);
            for (size_t j = 0; j < key_size; j++) {
                res[i * key_size + j] = res0[j];
            }
        }
        return res;
    }

    VT<unsigned> ApplyCircuitWithMeasure(const VT<BasicGate<T>> &circ, const ParameterResolver<T> &pr,
                                         const MST<size_t> &key_map) {
        auto key_size = key_map.size();
        VT<unsigned> res(key_size);
        for (auto &gate : circ) {
            if (gate.is_measure_) {
                auto collapse = ApplyMeasure(gate);
                res[key_map.at(gate.name_)] = collapse;
            } else if (gate.parameterized_) {
                ApplyGate(gate, pr);
            } else {
                ApplyGate(gate);
            }
        }
        return res;
    }

    void ApplyHamiltonian(const Hamiltonian<T> &ham) {
        Projectq::run();
        if (ham.how_to_ == ORIGIN) {
            Projectq::apply_qubit_operator(HCast<T>(ham.ham_), Projectq::ordering_);
        } else if (ham.how_to_ == BACKEND) {
            Projectq::vec_ = sparse::Csr_Dot_Vec<T, double>(ham.ham_sparse_main_, ham.ham_sparse_second_,
                                                            Projectq::vec_);
        } else {
            Projectq::vec_ = sparse::Csr_Dot_Vec<T, double>(ham.ham_sparse_main_, Projectq::vec_);
        }
    }

    VT<CT<T>> RightSizeGrad(calc_type *left_vec, calc_type *right_vec, const Hamiltonian<T> &ham,
                            const VT<BasicGate<T>> &circ, const VT<BasicGate<T>> &herm_circ,
                            const ParameterResolver<T> &pr, const MST<size_t> &p_map) {
        VT<CT<T>> f_g(p_map.size() + 1, 0);
        Projectq<T> sim_left = Projectq<T>(this->seed, n_qubits_, left_vec);
        sim_left.ApplyHamiltonian(ham);
        f_g[0] = ComplexInnerProduct<T, calc_type>(sim_left.vec_, right_vec, static_cast<Index>(len_));
        Projectq<T> sim_right = Projectq<T>(this->seed, n_qubits_, right_vec);
        Projectq<T> sim_right_tmp = Projectq<T>(this->seed, n_qubits_);
        for (size_t j = 0; j < circ.size(); j++) {
            if ((!herm_circ[j].parameterized_)
                || (herm_circ[j].params_.data_.size() == herm_circ[j].params_.no_grad_parameters_.size())) {
                if (herm_circ[j].parameterized_) {
                    sim_left.ApplyGate(herm_circ[j], pr, false);
                    sim_right.ApplyGate(herm_circ[j], pr, false);
                } else {
                    sim_left.ApplyGate(herm_circ[j]);
                    sim_right.ApplyGate(herm_circ[j]);
                }
            } else {
                sim_right.ApplyGate(herm_circ[j], pr, false);
                sim_right.run();
                sim_right_tmp.set_wavefunction(sim_right.vec_, ordering_);
                sim_right_tmp.ApplyGate(circ[circ.size() - j - 1], pr, true);
                sim_right_tmp.run();
                sim_left.run();
                CT<T> gi = 0;
                if (herm_circ[j].ctrl_qubits_.size() == 0) {
                    gi = ComplexInnerProduct<T, calc_type>(sim_left.vec_, sim_right_tmp.vec_, static_cast<Index>(len_));
                } else {
                    gi = ComplexInnerProductWithControl<T, calc_type>(sim_left.vec_, sim_right_tmp.vec_,
                                                                      static_cast<Index>(len_),
                                                                      GetControlMask(herm_circ[j].ctrl_qubits_));
                }
                for (auto &it : herm_circ[j].params_.GetRequiresGradParameters()) {
                    f_g[1 + p_map.at(it)] += circ[circ.size() - j - 1].params_.data_.at(it) * gi;
                }
                sim_left.ApplyGate(herm_circ[j], pr, false);
            }
        }
        return f_g;
    }

    CT<T> GetExpectation(const Hamiltonian<T> &ham) {
        Projectq<T> sim = Projectq<T>(this->seed, n_qubits_, vec_);
        sim.ApplyHamiltonian(ham);
        auto out = ComplexInnerProduct<T, calc_type>(sim.vec_, vec_, static_cast<Index>(len_));
        return out;
    }

    VT<VT<CT<T>>> HermitianMeasureWithGrad(const VT<Hamiltonian<T>> &hams, const VT<BasicGate<T>> &circ,
                                           const VT<BasicGate<T>> &herm_circ, const ParameterResolver<T> &pr,
                                           const MST<size_t> &p_map, size_t mea_threads) {
        auto n_hams = hams.size();
        auto n_params = pr.data_.size();
        VT<VT<CT<T>>> output;
        for (size_t i = 0; i < n_hams; i++) {
            output.push_back({});
            for (size_t j = 0; j < n_params + 1; j++) {
                output[i].push_back({0, 0});
            }
        }

        Projectq<T> sim = Projectq<T>(this->seed, n_qubits_, vec_);
        sim.ApplyCircuit(circ, pr);
        if (n_hams == 1) {
            auto f_g = sim.RightSizeGrad(sim.vec_, sim.vec_, hams[0], circ, herm_circ, pr, p_map);
            for (size_t g = 1; g < n_params + 1; g++) {
                f_g[g] += std::conj(f_g[g]);
            }
            output[0] = f_g;
        } else {
            size_t end = 0;
            size_t offset = n_hams / mea_threads;
            size_t left = n_hams % mea_threads;
            for (size_t i = 0; i < mea_threads; ++i) {
                size_t start = end;
                end = start + offset;
                if (i < left) {
                    end += 1;
                }

                auto task = [&, start, end]() {
                    for (size_t n = start; n < end; n++) {
                        auto f_g = sim.RightSizeGrad(sim.vec_, sim.vec_, hams[n], circ, herm_circ, pr, p_map);
                        for (size_t g = 1; g < n_params + 1; g++) {
                            f_g[g] += std::conj(f_g[g]);
                        }
                        output[n] = f_g;
                    }
                };
                Threadpool::GetInstance().Push(task);
            }
            while (true) {
                if (Threadpool::GetInstance().IsTasksEmpty() && Threadpool::GetInstance().GetIdle() == 0) {
                    break;
                }
            }
        }
        return output;
    }

    VT<VT<VT<CT<T>>>> HermitianMeasureWithGrad(const VT<Hamiltonian<T>> &hams, const VT<BasicGate<T>> &circ,
                                               const VT<BasicGate<T>> &herm_circ, const VVT<T> &enc_data,
                                               const VT<T> &ans_data, const VS &enc_name, const VS &ans_name,
                                               size_t batch_threads, size_t mea_threads) {
        auto n_hams = hams.size();
        auto n_prs = enc_data.size();
        auto n_params = enc_name.size() + ans_name.size();
        VT<VT<VT<CT<T>>>> output;
        for (size_t i = 0; i < n_prs; i++) {
            output.push_back({});
            for (size_t j = 0; j < n_hams; j++) {
                output[i].push_back({});
                for (size_t k = 0; k < n_params + 1; k++) {
                    output[i][j].push_back({0, 0});
                }
            }
        }
        MST<size_t> p_map;
        for (size_t i = 0; i < enc_name.size(); i++) {
            p_map[enc_name[i]] = i;
        }
        for (size_t i = 0; i < ans_name.size(); i++) {
            p_map[ans_name[i]] = i + enc_name.size();
        }

        if (n_prs == 1) {
            ParameterResolver<T> pr = ParameterResolver<T>();
            pr.SetItems(enc_name, enc_data[0]);
            pr.SetItems(ans_name, ans_data);
            output[0] = HermitianMeasureWithGrad(hams, circ, herm_circ, pr, p_map, mea_threads);
        } else {
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
                        ParameterResolver<T> pr = ParameterResolver<T>();
                        pr.SetItems(enc_name, enc_data[n]);
                        pr.SetItems(ans_name, ans_data);
                        auto f_g = HermitianMeasureWithGrad(hams, circ, herm_circ, pr, p_map, mea_threads);
                        output[n] = f_g;
                    }
                };
                tasks.emplace_back(task);
            }
            for (auto &t : tasks) {
                t.join();
            }
        }
        return output;
    }

    VT<VT<CT<T>>> NonHermitianMeasureWithGrad(const VT<Hamiltonian<T>> &hams, const VT<Hamiltonian<T>> &herm_hams,
                                              const VT<BasicGate<T>> &left_circ, const VT<BasicGate<T>> &herm_left_circ,
                                              const VT<BasicGate<T>> &right_circ,
                                              const VT<BasicGate<T>> &herm_right_circ, const ParameterResolver<T> &pr,
                                              const MST<size_t> &p_map, size_t mea_threads, const StateVector varphi) {
        auto n_hams = hams.size();
        auto n_params = pr.data_.size();
        VT<VT<CT<T>>> output;
        for (size_t i = 0; i < n_hams; i++) {
            output.push_back({});
            for (size_t j = 0; j < n_params + 1; j++) {
                output[i].push_back({0, 0});
            }
        }
        Projectq<T> sim = Projectq<T>(this->seed, n_qubits_, vec_);
        sim.ApplyCircuit(right_circ, pr);
        Projectq<T> sim2 = Projectq<T>(this->seed, n_qubits_, varphi);
        sim2.ApplyCircuit(left_circ, pr);
        if (n_hams == 1) {
            auto f_g1 = sim2.RightSizeGrad(sim.vec_, sim2.vec_, hams[0], left_circ, herm_left_circ, pr, p_map);
            auto f_g2 = sim.RightSizeGrad(sim2.vec_, sim.vec_, herm_hams[0], right_circ, herm_right_circ, pr, p_map);
            for (size_t g = 1; g < n_params + 1; g++) {
                f_g2[g] += std::conj(f_g1[g]);
            }
            output[0] = f_g2;
        } else {
            std::vector<std::thread> tasks;
            tasks.reserve(mea_threads);
            size_t end = 0;
            size_t offset = n_hams / mea_threads;
            size_t left = n_hams % mea_threads;
            for (size_t i = 0; i < mea_threads; i++) {
                size_t start = end;
                end = start + offset;
                if (i < left) {
                    end += 1;
                }
                auto task = [&, start, end]() {
                    for (size_t n = start; n < end; n++) {
                        auto f_g1 = sim2.RightSizeGrad(sim.vec_, sim2.vec_, hams[n], left_circ, herm_left_circ, pr,
                                                       p_map);
                        auto f_g2 = sim.RightSizeGrad(sim2.vec_, sim.vec_, herm_hams[n], right_circ, herm_right_circ,
                                                      pr, p_map);
                        for (size_t g = 1; g < n_params + 1; g++) {
                            f_g2[g] += std::conj(f_g1[g]);
                        }
                        output[n] = f_g2;
                    }
                };
                tasks.emplace_back(task);
            }
            for (auto &t : tasks) {
                t.join();
            }
        }
        return output;
    }

    VT<VT<VT<CT<T>>>> NonHermitianMeasureWithGrad(
        const VT<Hamiltonian<T>> &hams, const VT<Hamiltonian<T>> &herm_hams, const VT<BasicGate<T>> &left_circ,
        const VT<BasicGate<T>> &herm_left_circ, const VT<BasicGate<T>> &right_circ,
        const VT<BasicGate<T>> &herm_right_circ, const VVT<T> &enc_data, const VT<T> &ans_data, const VS &enc_name,
        const VS &ans_name, size_t batch_threads, size_t mea_threads, const Projectq<T> &simulator_left) {
        StateVector varphi = simulator_left.vec_;
        auto n_hams = hams.size();
        auto n_prs = enc_data.size();
        auto n_params = enc_name.size() + ans_name.size();
        VT<VT<VT<CT<T>>>> output;
        for (size_t i = 0; i < n_prs; i++) {
            output.push_back({});
            for (size_t j = 0; j < n_hams; j++) {
                output[i].push_back({});
                for (size_t k = 0; k < n_params + 1; k++) {
                    output[i][j].push_back({0, 0});
                }
            }
        }
        MST<size_t> p_map;
        for (size_t i = 0; i < enc_name.size(); i++) {
            p_map[enc_name[i]] = i;
        }
        for (size_t i = 0; i < ans_name.size(); i++) {
            p_map[ans_name[i]] = i + enc_name.size();
        }
        if (n_prs == 1) {
            ParameterResolver<T> pr = ParameterResolver<T>();
            pr.SetItems(enc_name, enc_data[0]);
            pr.SetItems(ans_name, ans_data);
            output[0] = NonHermitianMeasureWithGrad(hams, herm_hams, left_circ, herm_left_circ, right_circ,
                                                    herm_right_circ, pr, p_map, mea_threads, varphi);
        } else {
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
                        ParameterResolver<T> pr = ParameterResolver<T>();
                        pr.SetItems(enc_name, enc_data[n]);
                        pr.SetItems(ans_name, ans_data);
                        auto f_g = NonHermitianMeasureWithGrad(hams, herm_hams, left_circ, herm_left_circ, right_circ,
                                                               herm_right_circ, pr, p_map, mea_threads, varphi);
                        output[n] = f_g;
                    }
                };
                tasks.emplace_back(task);
            }
            for (auto &t : tasks) {
                t.join();
            }
        }
        return output;
    }

    void PrintInfo() const {
        std::cout << n_qubits_ << " qubits simulator with currently quantum state at:" << std::endl;
        for (unsigned i = 0; i < (len_ >> 1); i++) {
            std::cout << "(" << vec_[2 * i] << ", " << vec_[2 * i + 1] << ")" << std::endl;
        }
    }

    VVT<CT<calc_type>> GetCircuitMatrix(const VT<BasicGate<T>> &circ, const ParameterResolver<T> &pr) {
        VVT<CT<calc_type>> out((1 << n_qubits_));
        THRESHOLD_OMP_FOR(
            n_qubits_, nQubitTh, for (omp::idx_t i = 0; i < (1UL << n_qubits_); i++) {
                Projectq<T> sim = Projectq<T>(this->seed, n_qubits_);
                sim.vec_[0] = 0;
                sim.vec_[2 * i] = 1;
                sim.ApplyCircuit(circ, pr);
                out[i] = sim.cheat();
            })
        return out;
    }

    auto GetLen() const {
        return this->len_;
    }

    std::shared_ptr<Projectq<T>> Copy() {
        return std::make_shared<Projectq<T>>(this->seed, n_qubits_, vec_);
    }
};

template <typename T>
CT<T> InnerProduct(const Projectq<T> &bra, const Projectq<T> &ket) {
    auto res = ComplexInnerProduct<T, ::projectq::Simulator::calc_type>(bra.vec_, ket.vec_, bra.GetLen());
    return res;
}
}  // namespace projectq
}  // namespace mindquantum
#endif  // MINDQUANTUM_BACKENDS_PROJECTQ_PROJECTQ_H_
