//   Copyright 2023 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#include "device/mapping.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include <fmt/core.h>

#include "core/mq_base_types.hpp"
#include "ops/basic_gate.hpp"
#include "ops/gate_id.hpp"

namespace mindquantum::mapping {
// -----------------------------------------------------------------------------
VT<VT<int>> GetCircuitDAG(int n, const VT<Gate>& gates) {
    int m = gates.size();
    VT<int> last(n, -1);
    VT<VT<int>> DAG(m);

    for (int i = 0; i < gates.size(); ++i) {
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
    Index max_qubit = 0;
    for (size_t i = 0; i < gates.size(); i++) {
        auto gate = gates[i];
        if (gate->obj_qubits_.size() + gate->ctrl_qubits_.size() > 2) {
            throw std::runtime_error("Only works for gate with less than two qubits (control qubits included).");
        }
        if (gate->obj_qubits_.size() < 1) {
            throw std::runtime_error("Gate should be act on some qubits first.");
        }
        auto all_qubit = gate->obj_qubits_;
        all_qubit.insert(all_qubit.end(), gate->ctrl_qubits_.begin(), gate->ctrl_qubits_.end());
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
    for (int i = 0; i < pi.size(); ++i) {
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
    // return HBasic(F, pi);
    // return HLookAhead(F, E, pi);
    return std::max(decay[SWAP.first], decay[SWAP.second]) * HLookAhead(F, E, pi);
}

SABRE::SABRE(const VT<std::shared_ptr<BasicGate>>& circ, const std::shared_ptr<QubitsTopology>& coupling_graph) {
    auto tmp = GateToAbstractGate(circ);
    this->num_logical = tmp.first;
    this->gates = tmp.second;
    this->num_physical = coupling_graph->size();
    this->G = VT<VT<int>>(this->num_physical, VT<int>(0));
    for (auto id : coupling_graph->AllQubitID()) {
        auto nearby = (*coupling_graph)[id]->neighbour;
        this->G[id].insert(this->G[id].begin(), nearby.begin(), nearby.end());
    }

    // -----------------------------------------------------------------------------

    // get DAG and RDAG of logical circuit
    this->DAG = GetCircuitDAG(num_logical, gates);
    this->RDAG = VT<VT<int>>(this->DAG.size());
    for (int x; x < DAG.size(); ++x) {
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
    for (int i = 0; i < DAG.size(); ++i)
        for (int j : DAG[i])
            indeg[j]++;

    std::list<int> F;  // front layer
    for (int i = 0; i < DAG.size(); ++i)
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
            double min_score = __DBL_MAX__;
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
    for (int i = 0; i < pi.size(); ++i)
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
        if (g.type == "SWAP") {
            gate_info.push_back({-1, g.q1, g.q2});
        } else {
            gate_info.push_back({std::stoi(g.tag), g.q1, g.q2});
        }
    }
    return {gate_info, {initial_mapping, pi}};
}

inline void SABRE::SetParameters(double W, double delta1, double delta2) {
    this->W = W;
    this->delta1 = delta1;
    this->delta2 = delta2;
}
}  // namespace mindquantum::mapping
