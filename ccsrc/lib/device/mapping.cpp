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

#include "device/mapping.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>

#include <fmt/core.h>

#include "core/mq_base_types.h"
#include "device/topology.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"

namespace mindquantum::mapping {
// -----------------------------------------------------------------------------
VT<VT<int>> GetCircuitDAG(int n, const VT<Gate>& gates) {
    int m = gates.size();
    VT<int> last(n, -1);
    VT<VT<int>> DAG(m);

    for (int i = 0; i < static_cast<int>(gates.size()); ++i) {
        int q1 = gates[i].q1;
        int q2 = gates[i].q2;

        if (last[q1] != -1)
            DAG[last[q1]].push_back(i);
        last[q1] = i;

        if (gates[i].type == "CNOT" || gates[i].type == "SWAP") {
            if (last[q2] != -1)
                DAG[last[q2]].push_back(i);
            last[q2] = i;
        }
    }

    return DAG;
}

std::pair<qbit_t, VT<Gate>> GateToAbstractGate(const VT<std::shared_ptr<BasicGate>>& gates) {
    VT<Gate> abs_circ;
    qbit_t max_qubit = 0;
    for (size_t i = 0; i < gates.size(); i++) {
        auto gate = gates[i];
        auto all_qubit = gate->GetObjQubits();
        auto ctrl_qubits = gate->GetCtrlQubits();
        if (all_qubit.size() + ctrl_qubits.size() > 2) {
            throw std::runtime_error("Only works for gate with less than two qubits (control qubits included).");
        }
        if (all_qubit.size() < 1) {
            throw std::runtime_error("Gate should be act on some qubits first.");
        }
        all_qubit.insert(all_qubit.end(), ctrl_qubits.begin(), ctrl_qubits.end());
        auto local_max = *std::max_element(all_qubit.begin(), all_qubit.end());
        max_qubit = std::max(local_max, max_qubit);
        int q1 = all_qubit[0], q2;
        if (all_qubit.size() == 1) {
            q2 = q1;
        } else {
            q2 = all_qubit[1];
        }
        bool single_qubit = q1 == q2;
        std::string type = "CNOT";
        if (single_qubit) {
            type = "X";
        }
        abs_circ.emplace_back(Gate(type, q1, q2, std::to_string(i)));
    }
    return {max_qubit + 1, abs_circ};
}

// -----------------------------------------------------------------------------
int MQ_SABRE::CalGraphCenter(const VT<VT<int>>& graph) {
    int n = graph.size();
    int center_qubit = 0;
    int tempmin = INT16_MAX;
    for (int i = 0; i < n; i++) {
        int temp_sum = 0;
        for (int j = 0; j < n; j++) {
            temp_sum += graph[i][j];
        }
        if (temp_sum < tempmin) {
            tempmin = temp_sum;
            center_qubit = i;
        }
    }
    return center_qubit;
}

VT<int> MQ_SABRE::InitialMapping(const std::shared_ptr<QubitsTopology>& coupling_graph) {
    VT<int> layout(this->num_physical, -1);
    VT<int> Rlayout(this->num_physical, -1);
    int Qc = CalGraphCenter(this->D);
    int qc = CalGraphCenter(this->Gw);
    layout[qc] = Qc;
    Rlayout[Qc] = 1;
    std::queue<int> qcQueue;  // the queue of logic qubits
    std::queue<int> QcQueue;  // the queue of physical qubits
    qcQueue.push(qc);
    QcQueue.push(Qc);
    int flag = 1;  // the flag  whether it is the first mapping
    int count = 1;
    while (!qcQueue.empty() && !QcQueue.empty()) {
        int qi = qcQueue.front();
        int Qi = QcQueue.front();
        // judge whether the current logical bit matches the physical bit
        if (layout[qi] == -1) {
            layout[qi] = Qi;  //  if not ,then map them
        }
        // if it is the first mapping
        if (flag == 1) {
            flag = 0;
            // find all qubits associated with the current qubit qc
            for (int i = 0; i < this->num_logical; i++) {
                if (Gw[qc][i] != 0) {
                    qcQueue.push(i);
                }
            }
            VT<VT<int>> tempCandidatePysicalQubits(this->num_physical, VT<int>(2, 0));
            for (int i = 0; i < this->num_physical; i++) {
                if (i != Qi) {
                    tempCandidatePysicalQubits[i][0] = D[Qi][i];  // ****** 想想能不能优化
                    tempCandidatePysicalQubits[i][1] = i;
                } else {
                    tempCandidatePysicalQubits[i][0] = -1;
                    tempCandidatePysicalQubits[i][1] = i;
                }
            }
            sort(tempCandidatePysicalQubits.begin(), tempCandidatePysicalQubits.end(),
                 [](const VT<int> a, const VT<int> b) {
                     if (a[0] <= b[0])
                         return true;
                     return false;
                 });

            qcQueue.pop();
            QcQueue.pop();
            // find nearest physical qubits and its number equal with qcQueue.size
            for (int i = 0; i < qcQueue.size(); i++) {
                QcQueue.push(tempCandidatePysicalQubits[i + 1][1]);
                Rlayout[tempCandidatePysicalQubits[i + 1][1]] = 1;
            }
        } else {
            VT<int> SubTree;
            // find the subtree of the current logical bit association
            for (int i = 0; i < this->num_logical; i++) {
                if (Go[qi][i] != -1 && layout[i] == -1) {
                    SubTree.emplace_back(i);
                }
            }
            if (SubTree.size() > 0) {
                VT<VT<int>> tempCandidatePysicalQubits(this->num_physical, VT<int>(2, 0));
                for (int i = 0; i < this->num_physical; i++) {
                    if (i != Qi) {
                        tempCandidatePysicalQubits[i][0] = D[Qi][i];
                        tempCandidatePysicalQubits[i][1] = i;
                    } else {
                        tempCandidatePysicalQubits[i][0] = -1;
                        tempCandidatePysicalQubits[i][1] = i;
                    }
                }
                sort(tempCandidatePysicalQubits.begin(), tempCandidatePysicalQubits.end(),
                     [](const VT<int> a, const VT<int> b) {
                         if (a[0] <= b[0])
                             return true;
                         return false;
                     });
                // find nearest physical qubits and its number equal with SubTree.size
                for (int i = 0; i < SubTree.size(); i++) {
                    for (int j = 1; j < tempCandidatePysicalQubits.size(); j++) {
                        if (Rlayout[tempCandidatePysicalQubits[j][1]] == -1) {
                            qcQueue.push(SubTree[i]);
                            QcQueue.push(tempCandidatePysicalQubits[j][1]);
                            Rlayout[tempCandidatePysicalQubits[j][1]] = 1;
                            break;
                        }
                    }
                }
            }
            qcQueue.pop();
            QcQueue.pop();
        }
        count++;
    }
    int start_num_logical = 0;
    int start_num_physical = 0;
    // deal with qubits that don't match
    while (start_num_logical < this->num_logical) {
        if (layout[start_num_logical] == -1) {
            while (start_num_physical < this->num_physical && Rlayout[start_num_physical] != -1) {
                start_num_physical++;
            }
            layout[start_num_logical] = start_num_physical;
            Rlayout[start_num_physical] = start_num_logical;
        }
        start_num_logical++;
    }

    return layout;
}

std::list<int> MQ_SABRE::GetNextLayer(const std::list<int>& F, const VT<VT<int>>& DAG, VT<int>& indeg) {
    std::list<int> ret;
    for (int x : F) {
        for (int y : DAG[x]) {
            indeg[y]--;
            // y is CNOT gate
            if (gates[y].type == "CNOT") {
                if (indeg[y] == 0) {
                    ret.push_back(y);
                }
            } else {
                if (indeg[y] == 0) {
                    ret.push_back(y);
                }  // y is single gate
            }
        }
    }
    return ret;
}
std::list<int> MQ_SABRE::GetFLayer(std::list<std::pair<int, int>>& E) {
    std::list<int> res;
    if (E.size() > 0) {
        int flag = E.front().second;
        while (E.size() > 0 && flag == E.front().second) {
            res.push_back(E.front().first);
            E.pop_front();
        }
    }
    return res;
}

bool MQ_SABRE::IsExecutable(const VT<int>& pi, int g) const {
    if (gates[g].type == "CNOT") {
        int p = pi[gates[g].q1], q = pi[gates[g].q2];
        return std::any_of(G[p].begin(), G[p].end(), [&](int a) { return a == q; });
    } else {
        return true;
    }
}

std::set<std::pair<int, int>> MQ_SABRE::ObtainSWAPs(const std::list<int>& F, const VT<int>& pi) const {
    std::set<std::pair<int, int>> ret;
    for (int g : F) {
        int x = pi[gates[g].q1];
        int y = pi[gates[g].q2];
        for (int z : G[x]) {
            ret.insert({std::min(x, z), std::max(x, z)});
        }
        for (int z : G[y]) {
            ret.insert({std::min(y, z), std::max(y, z)});
        }
    }
    return ret;
}

double MQ_SABRE::HBasic(const std::list<int>& F, const VT<int>& pi) const {
    double sum = 0;
    for (int g : F) {
        int q1 = gates[g].q1;
        int q2 = gates[g].q2;
        sum += DM[pi[q1]][pi[q2]];
    }
    return sum;
}

std::pair<double, double> MQ_SABRE::HExtended(const std::list<std::pair<int, int>>& E, const VT<int>& tmppi,
                                              const VT<int>& pi) const {
    double sum = 0;
    double effectcost = 0;
    for (auto g : E) {
        int q1 = gates[g.first].q1;
        int q2 = gates[g.first].q2;
        sum += DM[tmppi[q1]][tmppi[q2]];
        effectcost += DM[tmppi[q1]][tmppi[q2]] - DM[pi[q1]][pi[q2]];
    }
    return std::pair<double, double>({sum, effectcost});
}

VT<int> MQ_SABRE::GetReversePi(const VT<int>& pi) const {
    VT<int> rpi(pi.size(), -1);
    for (int i = 0; i < static_cast<int>(pi.size()); ++i) {
        if (pi[i] != -1)
            rpi[pi[i]] = i;
    }
    return rpi;
}

VT<Gate> MQ_SABRE::HeuristicSearch(VT<int>& pi, const VT<VT<int>>& DAG) {
    VT<Gate> ans;
    int tot = 0;
    auto rpi = GetReversePi(pi);   // mapping from physical to logical
    VT<int> indeg(DAG.size(), 0);  // in-degree of DAG nodes
    int num_of_qubits = this->num_logical;
    for (int i = 0; i < static_cast<int>(DAG.size()); ++i) {
        for (int j : DAG[i]) {
            indeg[j]++;
        }
    }
    std::list<int> F;  // front layer
    std::list<std::pair<int, int>> E;
    // get first layer
    for (int i = 0; i < static_cast<int>(DAG.size()); ++i) {
        if (indeg[i] == 0) {
            F.push_back(i);
        }
    }
    int layer = 0;
    while (F.size() > 0) {
        ++layer;
        for (auto it = F.begin(); it != F.end(); ++it) {
            E.push_back({*it, layer});
        }
        F = GetNextLayer(F, DAG, indeg);
    }
    while (!E.empty() || !F.empty()) {
        if (F.empty()) {
            F = GetFLayer(E);
        }
        for (auto it = F.begin(); it != F.end();) {
            if (IsExecutable(pi, *it)) {
                int x = *it;
                if (gates[x].type == "CNOT") {
                    int p = gates[x].q1;
                    int q = gates[x].q2;
                    ans.push_back({"CNOT", pi[p], pi[q], gates[x].tag});
                } else {
                    int p = gates[x].q1;
                    ans.push_back({gates[x].type, pi[p], pi[p], gates[x].tag});
                }
                it = F.erase(it);
            } else {
                it++;
            }
        }
        if (!F.empty()) {
            double score = std::numeric_limits<double>::max();
            double effect;
            auto candidate_SWAPs = ObtainSWAPs(F, pi);
            std::pair<int, int> minSWAP;
            VT<int> tmppi;
            VT<int> tmprpi;
            double H_;
            int temp_num_of_qubits = num_of_qubits;
            for (auto SWAP : candidate_SWAPs) {
                int x = SWAP.first, y = SWAP.second;
                int p = rpi[x], q = rpi[y];
                tmppi = pi;
                tmprpi = rpi;
                if (p == -1) {
                    tmprpi[x] = temp_num_of_qubits;
                    tmppi[temp_num_of_qubits] = x;
                    p = temp_num_of_qubits;
                }
                if (q == -1) {
                    tmprpi[y] = temp_num_of_qubits;
                    tmppi[temp_num_of_qubits] = y;
                    q = temp_num_of_qubits;
                }
                std::swap(tmppi[p], tmppi[q]);
                double hbasic = HBasic(F, tmppi);
                auto extendAndeffect = HExtended(E, tmppi, pi);
                if (E.size() > 0) {
                    H_ = hbasic / static_cast<double>(F.size())
                         + this->W * extendAndeffect.first / static_cast<double>(E.size());
                } else {
                    H_ = hbasic / static_cast<double>(F.size());
                }
                if (H_ < score) {
                    score = H_;
                    minSWAP = SWAP;
                    effect = extendAndeffect.second;
                }
            }
            int tempflag = 0;
            if (effect < 0) {
                for (auto it = F.begin(); it != F.end();) {
                    if (IsExecutable(tmppi, *it)) {
                        int x = minSWAP.first, y = minSWAP.second;
                        int p = pi[gates[*it].q1], q = pi[gates[*it].q2];
                        // return VT<Gate>();
                        // add bridge gate
                        if ((p == x && D[q][y] == 1) || (p == y && D[q][x] == 1)) {
                            tempflag = 1;
                            if (p == x && D[q][y] == 1) {
                                ans.push_back({"CNOT", q, y, gates[*it].tag});
                                ans.push_back({"CNOT", y, p, gates[*it].tag});
                                ans.push_back({"CNOT", q, y, gates[*it].tag});
                                ans.push_back({"CNOT", y, p, gates[*it].tag});
                            } else if ((p == y && D[q][x] == 1)) {
                                ans.push_back({"CNOT", q, x, gates[*it].tag});
                                ans.push_back({"CNOT", x, p, gates[*it].tag});
                                ans.push_back({"CNOT", q, x, gates[*it].tag});
                                ans.push_back({"CNOT", x, p, gates[*it].tag});
                            }
                            it = F.erase(it);
                        }
                    }
                    it++;
                }
            }
            if (tempflag == 0) {
                // if not meet above conditions ,add swap gate
                int x = minSWAP.first, y = minSWAP.second;
                int p = rpi[x], q = rpi[y];
                if (p == -1) {
                    rpi[x] = num_of_qubits;
                    pi[num_of_qubits] = x;
                    p = num_of_qubits;
                    num_of_qubits++;
                }
                if (q == -1) {
                    rpi[y] = num_of_qubits;
                    pi[num_of_qubits] = y;
                    q = num_of_qubits;
                    num_of_qubits++;
                }
                std::swap(pi[p], pi[q]);
                std::swap(rpi[x], rpi[y]);
                ans.push_back({"SWAP", x, y, "SWAP" + std::to_string(++tot)});
            }
        }
    }
    return ans;
}

MQ_SABRE::MQ_SABRE(const VT<std::shared_ptr<BasicGate>>& circ, const std::shared_ptr<QubitsTopology>& coupling_graph,
                   const std::vector<std::pair<std::pair<int, int>, VT<double>>>& CnotErrrorRateAndGateLength) {
    auto tmp = GateToAbstractGate(circ);
    this->num_logical = tmp.first;
    this->gates = tmp.second;
    // Compress physical-qubit IDs to contiguous [0..num_physical-1]
    auto ids_set = coupling_graph->AllQubitID();
    VT<qbit_t> sorted_ids(ids_set.begin(), ids_set.end());
    std::sort(sorted_ids.begin(), sorted_ids.end());
    for (int i = 0; i < static_cast<int>(sorted_ids.size()); ++i) {
        old_to_new_[sorted_ids[i]] = i;
        new_to_old_.push_back(sorted_ids[i]);
    }
    this->num_physical = static_cast<int>(sorted_ids.size());
    if (this->num_logical > this->num_physical) {
        throw std::runtime_error("The number of logical qubits (" + std::to_string(this->num_logical)
                                 + ") cannot be greater than the number of physical qubits ("
                                 + std::to_string(this->num_physical) + ").");
    }

    this->CNOT_error_rate = VT<VT<double>>(this->num_physical, VT<double>(this->num_physical, 0.0));
    this->CNOT_gate_length = VT<VT<double>>(this->num_physical, VT<double>(this->num_physical, 0.0));
    // get cnot error rate matrix and cnot length matrix
    for (const auto& cnot_data : CnotErrrorRateAndGateLength) {
        int n_q1 = old_to_new_.at(cnot_data.first.first);
        int n_q2 = old_to_new_.at(cnot_data.first.second);
        CNOT_error_rate[n_q1][n_q2] = cnot_data.second[0];
        CNOT_gate_length[n_q1][n_q2] = cnot_data.second[1];
    }
    this->SWAP_success_rate = VT<VT<double>>(this->num_physical, VT<double>(this->num_physical));
    this->SWAP_gate_length = VT<VT<double>>(this->num_physical, VT<double>(this->num_physical));
    // get swap correct rate matrix and swap length matrix
    for (int i = 0; i < this->num_physical; i++) {
        for (int j = 0; j < this->num_physical; j++) {
            if (CNOT_error_rate[i][j] != 0) {
                this->SWAP_success_rate[i][j] = 1
                                                - (1 - (this->CNOT_error_rate[i][j])) * (1 - CNOT_error_rate[j][i])
                                                      * (1 - CNOT_error_rate[i][j]);
            }
            if (CNOT_gate_length[i][j] != 0) {
                SWAP_gate_length[i][j] = 2 * CNOT_gate_length[i][j] + CNOT_gate_length[j][i];
            }
        }
    }
    this->G = VT<VT<int>>(this->num_physical, VT<int>(0));
    for (auto old_id : sorted_ids) {
        int nid = old_to_new_[old_id];
        for (auto old_nb : (*coupling_graph)[old_id]->neighbour) {
            this->G[nid].push_back(old_to_new_[old_nb]);
        }
    }
    // get DAG of logical circuit
    this->DAG = GetCircuitDAG(num_logical, gates);

    // get D using Floyd algorithm
    {
        int n = num_physical;
        this->D = VT<VT<int>>(n, VT<int>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                D[i][j] = i == j ? 0 : 1e9;
        for (int i = 0; i < n; ++i)
            for (int j : G[i])
                D[i][j] = D[j][i] = 1;
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    D[i][j] = std::min(D[i][j], D[i][k] + D[k][j]);
    }
    // get Kesi by D
    {
        int n = num_physical;
        this->Kesi = VT<VT<double>>(n, VT<double>(n, 0.0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Kesi[i][j] = 1
                             - SWAP_success_rate[i][j] * SWAP_success_rate[j][i]
                                   * (std::max(SWAP_success_rate[i][j], SWAP_success_rate[j][i]));
            }
        }
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    Kesi[i][j] = std::min(Kesi[i][j], Kesi[i][k] + Kesi[k][j]);
    }
    // get T by D
    {
        int n = num_physical;
        this->T = VT<VT<double>>(n, VT<double>(n, 0.0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                T[i][j] = SWAP_gate_length[i][j] + SWAP_gate_length[j][i]
                          + std::min(SWAP_gate_length[i][j], SWAP_gate_length[j][i]);
            }
        }
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    T[i][j] = std::min(T[i][j], T[i][k] + T[k][j]);
    }
    // init Gw and Go
    {
        int n = num_logical;
        this->Go = VT<VT<int>>(n, VT<int>(n, -1));
        this->Gw = VT<VT<int>>(n, VT<int>(n, 0));
        this->Dw = VT<VT<int>>(n, VT<int>(n, 0));
        for (int i = 0; i < static_cast<int>(gates.size()); ++i) {
            if (gates[i].type == "CNOT") {
                int q1 = gates[i].q1;
                int q2 = gates[i].q2;
                Dw[q1][q2] += 1;
                Dw[q2][q1] += 1;
                Gw[q1][q2] += -1;
                Gw[q2][q1] += -1;
                if (Go[q1][q2] == -1) {
                    Go[q1][q2] = i;
                    Go[q2][q1] = i;
                }
            }
        }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (i != j && Dw[i][j] == 0)
                    Dw[i][j] = 1e9;
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    Dw[i][j] = std::min(Dw[i][j], Dw[i][k] + Dw[k][j]);
    }

    this->layout = this->InitialMapping(coupling_graph);  // 这个应该是物理量子特性，理应归属为类的成员
    // return;
}

std::pair<VT<VT<int>>, std::pair<VT<int>, VT<int>>> MQ_SABRE::Solve(double W, double alpha1, double alpha2,
                                                                    double alpha3) {
    this->SetParameters(W, alpha1, alpha2, alpha3);  // set parameters
    this->DM = VT<VT<double>>(num_physical, VT<double>(num_physical, 0.0));
    for (int i = 0; i < num_physical; i++) {
        for (int j = 0; j < num_physical; j++) {
            DM[i][j] = alpha1 * D[i][j] + alpha2 * Kesi[i][j] + alpha3 * T[i][j];
        }
    }
    auto initial_mapping = this->layout;                    // first mapping
    auto gates = HeuristicSearch(this->layout, this->DAG);  // final mapping

    VT<VT<int>> gate_info;
    for (auto& g : gates) {
        // translate back to original physical-qubit IDs
        qbit_t old_q1 = new_to_old_.at(g.q1);
        qbit_t old_q2 = new_to_old_.at(g.q2);
        if (g.type == "SWAP") {
            gate_info.push_back({-1, static_cast<int>(old_q1), static_cast<int>(old_q2)});
        } else {
            gate_info.push_back({std::stoi(g.tag), static_cast<int>(old_q1), static_cast<int>(old_q2)});
        }
    }
    // remap initial and final logical->physical mappings back to original IDs
    VT<int> init_map_remap(initial_mapping.size());
    for (size_t i = 0; i < initial_mapping.size(); i++) {
        if (initial_mapping[i] != -1) {
            init_map_remap[i] = static_cast<int>(new_to_old_.at(initial_mapping[i]));
        } else {
            init_map_remap[i] = -1;
        }
    }
    VT<int> final_map_remap(this->layout.size());
    for (size_t i = 0; i < this->layout.size(); i++) {
        if (this->layout[i] != -1) {
            final_map_remap[i] = static_cast<int>(new_to_old_.at(this->layout[i]));
        } else {
            final_map_remap[i] = -1;
        }
    }
    return {gate_info, {init_map_remap, final_map_remap}};
}

inline void MQ_SABRE::SetParameters(double W, double alpha1, double alpha2, double alpha3) {
    this->W = W;
    this->alpha1 = alpha1;
    this->alpha2 = alpha2;
    this->alpha3 = alpha3;
}
// -----------------------------------------------------------------------------
bool SABRE::IsExecutable(const VT<int>& pi, int g) const {
    if (gates[g].type == "CNOT") {
        int p = pi[gates[g].q1], q = pi[gates[g].q2];
        return std::any_of(G[p].begin(), G[p].end(), [&](int a) { return a == q; });
    } else {
        return true;
    }
}

VT<int> SABRE::GetReversePi(const VT<int>& pi) const {
    VT<int> rpi(pi.size());
    for (int i = 0; i < static_cast<int>(pi.size()); ++i) {
        rpi[pi[i]] = i;
    }
    return rpi;
}

std::set<std::pair<int, int>> SABRE::ObtainSWAPs(const std::list<int>& F, const VT<int>& pi) const {
    std::set<std::pair<int, int>> ret;
    for (int g : F) {
        int x = pi[gates[g].q1];
        int y = pi[gates[g].q2];
        for (int z : G[x])
            ret.insert({std::min(x, z), std::max(x, z)});
        for (int z : G[y])
            ret.insert({std::min(y, z), std::max(y, z)});
    }
    return ret;
}

std::list<int> SABRE::GetNextLayer(const std::list<int>& F, const VT<VT<int>>& DAG, const VT<int>& indeg) const {
    VT<int> tmp_deg = indeg;
    std::list<int> ret;
    for (int x : F) {
        for (int y : DAG[x]) {
            tmp_deg[y]--;
            if (gates[y].type == "CNOT") {  // y is CNOT gate
                if (tmp_deg[y] == 0)
                    ret.push_back(y);
            } else {                    // y is single gate
                for (int z : DAG[y]) {  // find following gate
                    tmp_deg[z]--;
                    if (tmp_deg[z] == 0)
                        ret.push_back(z);
                }
            }
        }
    }
    return ret;
}

std::list<int> SABRE::GetExtendedSet(const std::list<int>& F, const VT<VT<int>>& DAG, const VT<int>& indeg) const {
    return GetNextLayer(F, DAG, indeg);
}

double SABRE::HBasic(const std::list<int>& F, const VT<int>& pi) const {
    int sum = 0;
    for (int g : F) {
        int q1 = gates[g].q1;
        int q2 = gates[g].q2;
        sum += D[pi[q1]][pi[q2]];
    }
    return sum;
}

double SABRE::HLookAhead(const std::list<int>& F, const std::list<int>& E, const VT<int>& pi) const {
    double s1 = HBasic(F, pi) / static_cast<double>(F.size());
    if (E.size() == 0) {
        return s1;
    } else {
        double s2 = HBasic(E, pi) / static_cast<double>(E.size());
        return s1 + W * s2;  // where 0 <= W <= 1 is a parameter
    }
}

double SABRE::H(const std::list<int>& F, const std::list<int>& E, const VT<int>& pi, const std::pair<int, int>& SWAP,
                const VT<double>& decay) const {
    return std::max(decay[SWAP.first], decay[SWAP.second]) * HLookAhead(F, E, pi);
}

SABRE::SABRE(const VT<std::shared_ptr<BasicGate>>& circ, const std::shared_ptr<QubitsTopology>& coupling_graph) {
    auto tmp = GateToAbstractGate(circ);
    this->num_logical = tmp.first;
    this->gates = tmp.second;
    // Compress physical-qubit IDs to contiguous [0..num_physical-1]
    auto ids_set = coupling_graph->AllQubitID();
    VT<qbit_t> sorted_ids(ids_set.begin(), ids_set.end());
    std::sort(sorted_ids.begin(), sorted_ids.end());
    for (int i = 0; i < static_cast<int>(sorted_ids.size()); ++i) {
        old_to_new_[sorted_ids[i]] = i;
        new_to_old_.push_back(sorted_ids[i]);
    }
    this->num_physical = static_cast<int>(sorted_ids.size());
    if (this->num_logical > this->num_physical) {
        throw std::runtime_error("The number of logical qubits (" + std::to_string(this->num_logical)
                                 + ") cannot be greater than the number of physical qubits ("
                                 + std::to_string(this->num_physical) + ").");
    }
    this->G = VT<VT<int>>(num_physical, VT<int>(0));
    for (auto old_id : sorted_ids) {
        int nid = old_to_new_[old_id];
        for (auto old_nb : (*coupling_graph)[old_id]->neighbour) {
            this->G[nid].push_back(old_to_new_[old_nb]);
        }
    }
    // -----------------------------------------------------------------------------

    // get DAG and RDAG of logical circuit
    this->DAG = GetCircuitDAG(num_logical, gates);
    this->RDAG = VT<VT<int>>(this->DAG.size());
    for (int x = 0; x < static_cast<int>(DAG.size()); ++x) {
        for (int y : DAG[x])
            RDAG[y].push_back(x);
    }

    // get D using Floyd algorithm
    {
        int n = num_physical;
        this->D = VT<VT<int>>(n, VT<int>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                D[i][j] = i == j ? 0 : 1e9;
        for (int i = 0; i < n; ++i)
            for (int j : G[i])
                D[i][j] = D[j][i] = 1;
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    D[i][j] = std::min(D[i][j], D[i][k] + D[k][j]);
    }
}

VT<Gate> SABRE::HeuristicSearch(VT<int>& pi, const VT<VT<int>>& DAG) {
    VT<Gate> ans;  // physical circuit
    int tot = 0;   // total number of additional SWAP gates

    auto rpi = GetReversePi(pi);     // mapping from physical to logical
    VT<double> decay(pi.size(), 1);  // decay of logical qubits

    VT<int> indeg(DAG.size(), 0);  // in-degree of DAG nodes
    for (int i = 0; i < static_cast<int>(DAG.size()); ++i)
        for (int j : DAG[i])
            indeg[j]++;

    std::list<int> F;  // front layer
    for (int i = 0; i < static_cast<int>(DAG.size()); ++i)
        if (indeg[i] == 0)
            F.push_back(i);

    while (!F.empty()) {
        VT<int> executable_list;
        // find all executable gates in F
        for (auto it = F.begin(); it != F.end(); ++it) {
            if (IsExecutable(pi, *it)) {
                executable_list.push_back(*it);
            }
        }

        if (!executable_list.empty()) {  // execute all executable gates
            for (auto it = F.begin(); it != F.end();) {
                if (IsExecutable(pi, *it)) {
                    int x = *it;
                    if (gates[x].type == "CNOT") {
                        int p = gates[x].q1;
                        int q = gates[x].q2;
                        double tmp = std::max(decay[p], decay[q]);
                        decay[p] = decay[q] = tmp + delta2;
                        ans.push_back({"CNOT", pi[p], pi[q], gates[x].tag});
                    } else {
                        int p = gates[x].q1;
                        decay[p] += delta1;
                        ans.push_back({gates[x].type, pi[p], pi[p], gates[x].tag});
                    }

                    for (int y : DAG[x]) {
                        --indeg[y];
                        if (indeg[y] == 0)
                            F.push_back(y);
                    }
                    it = F.erase(it);
                } else {
                    ++it;
                }
            }
        } else {  // If there is no executable gate, try to SWAP
            auto candidate_SWAPs = ObtainSWAPs(F, pi);
            auto E = GetExtendedSet(F, DAG, indeg);
            // find the SWAP with minimal H-score
            double min_score = std::numeric_limits<double>::max();
            std::pair<int, int> min_SWAP;
            for (auto SWAP : candidate_SWAPs) {
                int x = SWAP.first, y = SWAP.second;
                int p = rpi[x], q = rpi[y];

                auto tmp = pi;
                std::swap(tmp[p], tmp[q]);
                double score = H(F, E, tmp, {p, q}, decay);
                if (score < min_score) {
                    min_score = score;
                    min_SWAP = SWAP;
                }
            }

            int x = min_SWAP.first, y = min_SWAP.second;
            int p = rpi[x], q = rpi[y];
            std::swap(pi[p], pi[q]);
            std::swap(rpi[x], rpi[y]);
            ans.push_back({"SWAP", x, y, "SWAP" + std::to_string(++tot)});

            double tmp = std::max(decay[p], decay[q]);
            decay[p] = decay[q] = tmp + delta2 * 3;
        }
    }
    return ans;
}

void SABRE::IterOneTurn(VT<int>& pi) {
    HeuristicSearch(pi, this->DAG);   // using original circuit to update
    HeuristicSearch(pi, this->RDAG);  // using reversed circuit to update
}

std::pair<VT<VT<int>>, std::pair<VT<int>, VT<int>>> SABRE::Solve(int iter_num, double W, double delta1, double delta2) {
    this->SetParameters(W, delta1, delta2);

    // generate random initial mapping
    VT<int> pi(this->num_physical);
    for (int i = 0; i < static_cast<int>(pi.size()); ++i)
        pi[i] = i;

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto engine = std::default_random_engine(seed);
    shuffle(pi.begin(), pi.end(), engine);

    // iterate to update initial mapping
    for (int t = 0; t < iter_num; ++t) {
        IterOneTurn(pi);
    }
    auto initial_mapping = pi;
    auto gs = HeuristicSearch(pi, this->DAG);
    VT<VT<int>> gate_info;
    for (auto& g : gs) {
        // translate back to original physical-qubit IDs
        qbit_t old_q1 = new_to_old_[g.q1];
        qbit_t old_q2 = new_to_old_[g.q2];
        if (g.type == "SWAP") {
            gate_info.push_back({-1, static_cast<int>(old_q1), static_cast<int>(old_q2)});
        } else {
            gate_info.push_back({std::stoi(g.tag), static_cast<int>(old_q1), static_cast<int>(old_q2)});
        }
    }
    // remap initial and final logical->physical mappings back to original IDs
    VT<int> init_map_remap;
    VT<int> final_map_remap;
    init_map_remap.reserve(initial_mapping.size());
    std::transform(initial_mapping.cbegin(), initial_mapping.cend(), std::back_inserter(init_map_remap),
                   [this](int v) { return static_cast<int>(new_to_old_[v]); });
    final_map_remap.reserve(pi.size());
    std::transform(pi.cbegin(), pi.cend(), std::back_inserter(final_map_remap),
                   [this](int v) { return static_cast<int>(new_to_old_[v]); });
    return {gate_info, {init_map_remap, final_map_remap}};
}

inline void SABRE::SetParameters(double W, double delta1, double delta2) {
    this->W = W;
    this->delta1 = delta1;
    this->delta2 = delta2;
}
}  // namespace mindquantum::mapping
