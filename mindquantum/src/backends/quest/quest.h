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
#ifndef MINDQUANTUM_BACKENDS_QUEST_QUEST_H_
#define MINDQUANTUM_BACKENDS_QUEST_QUEST_H_
#include <string>

#include "QuEST.h"
#include "QuEST_internal.h"
#include "backends/quest/quest_utils.h"
#include "core/utils.h"
#include "gate/basic_gate.h"
#include "hamiltonian/hamiltonian.h"
#include "pr/parameter_resolver.h"

namespace mindquantum {
namespace quest {
template <typename T>
struct Quest {
    int n_qubits_;
    QuESTEnv env;
    Qureg qubits;
    explicit Quest(int n) : n_qubits_(n), env(createQuESTEnv()), qubits(createQureg(n, env)) {
        initZeroState(qubits);
    }
    Quest(int n, Qureg ref_qureg) : n_qubits_(n), env(createQuESTEnv()), qubits(createQureg(n, env)) {
        cloneQureg(qubits, ref_qureg);
    }
    Quest() : n_qubits_(1), env(createQuESTEnv()), qubits(createQureg(n_qubits_, env)) {
        initZeroState(qubits);
    }
    ~Quest() {
        destroyQureg(qubits, env);
        destroyQuESTEnv(env);
    }
    void InitializeSimulator() {
        initZeroState(qubits);
    }
    void PrintInfo() {
        reportQuregParams(qubits);
        reportQuESTEnv(env);
    }
    VT<CT<T>> GetVec() {
        VT<CT<T>> result;
        for (size_t i = 0; i < (1UL << n_qubits_); i++) {
            result.push_back({getRealAmp(qubits, i), getImagAmp(qubits, i)});
        }
        return result;
    }
    void ApplyGate(const BasicGate<T> &gate) {
        if (gate.ctrl_qubits_.size() == 0) {  // no control
            if (gate.name_ == gX) {
                pauliX(qubits, gate.obj_qubits_[0]);
            } else if (gate.name_ == gY) {
                pauliY(qubits, gate.obj_qubits_[0]);
            } else if (gate.name_ == gZ) {
                pauliZ(qubits, gate.obj_qubits_[0]);
            } else if (gate.name_ == gH) {
                hadamard(qubits, gate.obj_qubits_[0]);
            } else if (gate.name_ == gT) {
                tGate(qubits, gate.obj_qubits_[0]);
            } else if (gate.name_ == gS) {
                sGate(qubits, gate.obj_qubits_[0]);
            } else if (gate.name_ == gCNOT) {
                controlledNot(qubits, gate.obj_qubits_[1], gate.obj_qubits_[0]);
            } else if (gate.name_ == gSWAP) {
                swapGate(qubits, gate.obj_qubits_[0], gate.obj_qubits_[1]);
            } else if (gate.name_ == gRX) {
                rotateX(qubits, gate.obj_qubits_[0], static_cast<qreal>(gate.applied_value_));
            } else if (gate.name_ == gRY) {
                rotateY(qubits, gate.obj_qubits_[0], static_cast<qreal>(gate.applied_value_));
            } else if (gate.name_ == gRZ) {
                rotateZ(qubits, gate.obj_qubits_[0], static_cast<qreal>(gate.applied_value_));
            } else if (gate.name_ == gPS) {
                phaseShift(qubits, gate.obj_qubits_[0], static_cast<qreal>(gate.applied_value_));
            } else {
                auto obj = Vec2Intptr(gate.obj_qubits_);
                auto m = Dim2Matrix2ComplexMatrixN<T>(gate.base_matrix_, gate.obj_qubits_.size());
                multiQubitUnitary(qubits, obj, static_cast<int>(gate.obj_qubits_.size()), m);
                if (obj != nullptr) {
                    free(obj);
                }
                destroyComplexMatrixN(m);
            }
        } else if (gate.ctrl_qubits_.size() == 1) {
            if (gate.name_ == gX) {
                controlledNot(qubits, gate.ctrl_qubits_[0], gate.obj_qubits_[0]);
            } else {
                auto obj = Vec2Intptr(gate.obj_qubits_);
                auto m = Dim2Matrix2ComplexMatrixN<T>(gate.base_matrix_, gate.obj_qubits_.size());
                controlledMultiQubitUnitary(qubits, gate.ctrl_qubits_[0], obj,
                                            static_cast<int>(gate.obj_qubits_.size()), m);
                if (obj != nullptr) {
                    free(obj);
                }
                destroyComplexMatrixN(m);
            }
        } else {
            auto ctrl = Vec2Intptr(gate.ctrl_qubits_);
            auto obj = Vec2Intptr(gate.obj_qubits_);
            auto m = Dim2Matrix2ComplexMatrixN<T>(gate.base_matrix_, gate.obj_qubits_.size());
            int nctrl = static_cast<int>(gate.ctrl_qubits_.size());
            int nobj = static_cast<int>(gate.obj_qubits_.size());
            multiControlledMultiQubitUnitary(qubits, ctrl, nctrl, obj, nobj, m);
            destroyComplexMatrixN(m);
            if (obj != nullptr) {
                free(obj);
            }
            if (ctrl != nullptr) {
                free(ctrl);
            }
        }
    }
    void ApplyGate(const BasicGate<T> &gate, const ParameterResolver<T> &pr, bool diff = false) {
        T theta = LinearCombine(pr, gate.params_);
        if (diff) {
            auto ctrl = Vec2Intptr(gate.ctrl_qubits_);
            auto obj = Vec2Intptr(gate.obj_qubits_);
            int nctrl = static_cast<int>(gate.ctrl_qubits_.size());
            int nobj = static_cast<int>(gate.obj_qubits_.size());
            if (nctrl == 0 && nobj == 1) {
                auto m = Dim2Matrix2ComplexMatrix2<T>(gate.param_diff_matrix_(theta));
                applyMatrix2(qubits, gate.obj_qubits_[0], m);
            } else if (nctrl == 0 && nobj == 2) {
                auto m = Dim2Matrix2ComplexMatrix4<T>(gate.param_diff_matrix_(theta));
                applyMatrix4(qubits, gate.obj_qubits_[0], gate.obj_qubits_[1], m);
            } else if (nctrl != 0) {
                auto m = Dim2Matrix2ComplexMatrixN<T>(gate.param_diff_matrix_(theta), gate.obj_qubits_.size());
                applyMultiControlledMatrixN(qubits, ctrl, nctrl, obj, nobj, m);
                destroyComplexMatrixN(m);
            } else {
                auto m = Dim2Matrix2ComplexMatrixN<T>(gate.param_diff_matrix_(theta), gate.obj_qubits_.size());
                applyMatrixN(qubits, obj, nobj, m);
                destroyComplexMatrixN(m);
            }
            if (obj != nullptr) {
                free(obj);
            }
            if (ctrl != nullptr) {
                free(ctrl);
            }
        } else {
            BasicGate<T> gate_tmp = gate;
            gate_tmp.ApplyValue(theta);
            ApplyGate(gate_tmp);
        }
    }
    void ApplyCircuit(const VT<BasicGate<T>> &circ) {
        for (auto &gate : circ) {
            Quest::ApplyGate(gate);
        }
    }
    unsigned ApplyMeasure(const BasicGate<T> &gate) {
    }

    VT<unsigned> Sampling(const VT<BasicGate<T>> &circ, const ParameterResolver<T> &pr, size_t shots,
                          const MST<size_t> &key_map, unsigned seed) {
    }

    void ApplyCircuit(const VT<BasicGate<T>> &circ, const ParameterResolver<T> &pr) {
        for (auto &gate : circ) {
            if (gate.parameterized_) {
                Quest::ApplyGate(gate, pr);
            } else {
                Quest::ApplyGate(gate);
            }
        }
    }

    void ApplyHamiltonian(const Hamiltonian<T> &ham) {
        if (ham.how_to_ == ORIGIN) {
            Qureg qureg = createQureg(n_qubits_, env);
            Qureg ori = createQureg(n_qubits_, env);
            cloneQureg(ori, qubits);
            initBlankState(qubits);
            for (size_t i = 0; i < ham.ham_.size(); i++) {
                cloneQureg(qureg, ori);
                for (size_t j = 0; j < ham.ham_[i].first.size(); j++) {
                    if (ham.ham_[i].first[j].second == 'X') {
                        statevec_pauliX(qureg, ham.ham_[i].first[j].first);
                    }
                    if (ham.ham_[i].first[j].second == 'Y') {
                        statevec_pauliY(qureg, ham.ham_[i].first[j].first);
                    }
                    if (ham.ham_[i].first[j].second == 'Z') {
                        statevec_pauliZ(qureg, ham.ham_[i].first[j].first);
                    }
                }
                Complex coef = (Complex){.real = ham.ham_[i].second, .imag = 0};
                Complex iden = (Complex){.real = 1, .imag = 0};
                Complex zero = (Complex){.real = 0, .imag = 0};
                setWeightedQureg(coef, qureg, iden, qubits, zero, qubits);
            }
            destroyQureg(qureg, env);
            destroyQureg(ori, env);
        } else if (ham.how_to_ == BACKEND) {
            Csr_Dot_Vec<T>(ham.ham_sparse_main_, ham.ham_sparse_second_, qubits);
        } else {
            Csr_Dot_Vec<T>(ham.ham_sparse_main_, qubits);
        }
    }

    CT<qreal> GetExpectation(const Hamiltonian<T> &ham) {
        // auto quest_ham = HCast<T>(ham, n_qubits_);
        Quest<T> sim = Quest<T>(n_qubits_);
        cloneQureg(sim.qubits, qubits);
        sim.ApplyHamiltonian(ham);
        CT<qreal> value;
        if (qubits.isDensityMatrix) {
            value = {calcTotalProb(sim.qubits), 0};
        } else {
            value = Complex2Complex<T>(calcInnerProduct(qubits, sim.qubits));
        }

        return value;
    }

    VT<CT<T>> RightSizeGrad(Qureg left_vec, Qureg right_vec, const Hamiltonian<T> &ham, const VT<BasicGate<T>> &circ,
                            const VT<BasicGate<T>> &herm_circ, const ParameterResolver<T> &pr,
                            const MST<size_t> &p_map) {
        VT<CT<T>> f_g(p_map.size() + 1, 0);
        Quest<T> sim_left = Quest<T>(n_qubits_, left_vec);
        sim_left.ApplyHamiltonian(ham);
        f_g[0] = Complex2Complex<T>(calcInnerProduct(sim_left.qubits, right_vec));
        Quest<T> sim_right = Quest<T>(n_qubits_, right_vec);
        Quest<T> sim_right_tmp = Quest<T>(n_qubits_);
        for (size_t j = 0; j < circ.size(); j++) {
            if ((!herm_circ[j].parameterized_) || (herm_circ[j].params_.requires_grad_parameters_.size() == 0)) {
                if (herm_circ[j].parameterized_) {
                    sim_left.ApplyGate(herm_circ[j], pr, false);
                    sim_right.ApplyGate(herm_circ[j], pr, false);
                } else {
                    sim_left.ApplyGate(herm_circ[j]);
                    sim_right.ApplyGate(herm_circ[j]);
                }
            } else {
                sim_right.ApplyGate(herm_circ[j], pr, false);
                cloneQureg(sim_right_tmp.qubits, sim_right.qubits);
                sim_right_tmp.ApplyGate(circ[circ.size() - j - 1], pr, true);
                CT<T> gi = 0;
                // if (herm_circ[j].ctrl_qubits_.size() == 0) {
                gi = Complex2Complex<T>(calcInnerProduct(sim_left.qubits, sim_right_tmp.qubits));
                // } else {
                //   gi = ComplexInnerProductWithControl<T, calc_type>(sim_left.vec_,
                //   sim_right_tmp.vec_, static_cast<Index>(len_),
                //                                                     GetControlMask(herm_circ[j].ctrl_qubits_));
                // }
                for (auto &it : herm_circ[j].params_.requires_grad_parameters_) {
                    f_g[1 + p_map.at(it)] -= herm_circ[j].params_.data_.at(it) * gi;
                }
                sim_left.ApplyGate(herm_circ[j], pr, false);
            }
        }
        return f_g;
    }

    VT<VT<CT<T>>> HermitianMeasureWithGrad(const VT<Hamiltonian<T>> &hams, const VT<BasicGate<T>> &circ,
                                           const VT<BasicGate<T>> &herm_circ, const ParameterResolver<T> &pr,
                                           const VT<std::string> &params_order, size_t mea_threads) {
        auto n_hams = hams.size();
        auto n_params = pr.data_.size();
        MST<size_t> p_map;
        for (size_t i = 0; i < params_order.size(); i++) {
            p_map[params_order[i]] = i;
        }
        VT<VT<CT<T>>> output(n_hams);
        Quest<T> sim = Quest<T>(n_qubits_, qubits);
        auto t0 = NOW();
        sim.ApplyCircuit(circ, pr);
        auto t1 = NOW();
        std::cout << "evolution time " << TimeDuration(t0, t1) << std::endl;

        // #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_hams; i++) {
            auto t2 = NOW();
            auto f_g = sim.RightSizeGrad(sim.qubits, sim.qubits, hams[i], circ, herm_circ, pr, p_map);
            auto t3 = NOW();
            std::cout << "grad " << TimeDuration(t2, t3) << std::endl;
            for (size_t g = 1; g < n_params + 1; g++) {
                f_g[g] += std::conj(f_g[g]);
            }
            output[i] = f_g;
        }
        return output;
    }

    VT<VT<VT<CT<T>>>> HermitianMeasureWithGrad(const VT<Hamiltonian<T>> &hams, const VT<BasicGate<T>> &circ,
                                               const VT<BasicGate<T>> &herm_circ, const VT<ParameterResolver<T>> &prs,
                                               const VT<std::string> &params_order, size_t batch_threads,
                                               size_t mea_threads) {
        // auto n_hams = hams.size();
        auto n_prs = prs.size();
        // auto n_params = prs[0].data_.size();
        VT<VT<VT<CT<T>>>> output(n_prs);
        // #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_prs; i++) {
            auto f_g = HermitianMeasureWithGrad(hams, circ, herm_circ, prs[i], params_order, mea_threads);
            output[i] = f_g;
        }
        return output;
    }
};
}  // namespace quest
}  // namespace mindquantum
#endif
