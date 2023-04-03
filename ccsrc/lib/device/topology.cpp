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

#include "device/topology.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace mindquantum::mapping {
// =============================================================================

QubitNode::QubitNode(qbit_t id, std::string color, double poi_x, double poi_y)
    : id(id), color(color), poi_x(poi_x), poi_y(poi_y) {
}

void QubitNode::SetPoi(double poi_x, double poi_y) {
    this->poi_x = poi_x;
    this->poi_y = poi_y;
}

QNodePtr QubitNode::operator<<(const QNodePtr& other) {
    if (this->id == other->id) {
        throw std::runtime_error("Cannot connect itself.");
    }
    this->neighbour.insert(other->id);
    other->neighbour.insert(this->id);
    return shared_from_this();
}

QNodePtr QubitNode::operator>>(const QNodePtr& other) {
    if (this->id == other->id) {
        throw std::runtime_error("Cannot connect itself.");
    }
    this->neighbour.insert(other->id);
    other->neighbour.insert(this->id);
    return other;
}

QNodePtr QubitNode::operator<(const QNodePtr& other) {
    if (this->id == other->id) {
        throw std::runtime_error("Cannot disconnect itself.");
    }
    this->neighbour.erase(other->id);
    other->neighbour.erase(this->id);
    return shared_from_this();
}

QNodePtr QubitNode::operator>(const QNodePtr& other) {
    if (this->id == other->id) {
        throw std::runtime_error("Cannot disconnect itself.");
    }
    this->neighbour.erase(other->id);
    other->neighbour.erase(this->id);
    return other;
}

QubitNode::operator qbit_t() {
    return this->id;
}

// =============================================================================

QubitsTopology::QubitsTopology(const VT<QNodePtr>& qubits) {
    for (auto& qubit : qubits) {
        auto [it, succeed] = this->qubits.insert(std::unordered_map<qbit_t, QNodePtr>::value_type(qubit->id, qubit));
        if (!succeed) {
            throw std::runtime_error("qubit with id " + std::to_string(qubit->id) + " already exists.");
        }
    }
}

QNodePtr QubitsTopology::operator[](qbit_t id) {
    if (!(this->qubits.count(id) > 0)) {
        throw std::runtime_error("qubit with id " + std::to_string(id) + " not in this topology.");
    }
    return this->qubits.at(id);
}

int QubitsTopology::size() {
    return this->qubits.size();
}

int QubitsTopology::NEdges() {
    int n_edges = 0;
    for (auto& [id, qubit] : this->qubits) {
        n_edges += qubit->neighbour.size();
    }
    return n_edges / 2;
}

std::set<qbit_t> QubitsTopology::AllQubitID() {
    std::set<qbit_t> all_qubit_id;
    for (auto& [id, qubit] : this->qubits) {
        all_qubit_id.insert(id);
    }
    return all_qubit_id;
}

void QubitsTopology::SetPosition(qbit_t id, double poi_x, double poi_y) {
    (*this)[id]->SetPoi(poi_x, poi_y);
}

void QubitsTopology::SetPosition(std::map<qbit_t, std::pair<double, double>> positions) {
    for (auto& [id, poi] : positions) {
        this->SetPosition(id, poi.first, poi.second);
    }
}

void QubitsTopology::SetColor(qbit_t id, std::string color) {
    (*this)[id]->color = color;
}

void QubitsTopology::SetColor(std::map<qbit_t, std::string> colors) {
    for (auto& [id, color] : colors) {
        this->SetColor(id, color);
    }
}

std::set<std::pair<qbit_t, qbit_t>> QubitsTopology::EdgesWithID() {
    std::set<std::pair<qbit_t, qbit_t>> out;
    for (auto& [id_1, qubit] : this->qubits) {
        for (auto& id_2 : qubit->neighbour) {
            if (id_1 > id_2) {
                out.insert({id_2, id_1});
            } else {
                out.insert({id_1, id_2});
            }
        }
    }
    return out;
}

std::set<std::pair<std::pair<double, double>, std::pair<double, double>>> QubitsTopology::EdgesWithPoi() {
    std::set<std::pair<std::pair<double, double>, std::pair<double, double>>> out;
    for (auto& [id_1, id_2] : this->EdgesWithID()) {
        auto qubit_1 = (*this)[id_1];
        auto qubit_2 = (*this)[id_2];
        out.insert({{qubit_1->poi_x, qubit_1->poi_y}, {qubit_2->poi_x, qubit_2->poi_y}});
    }
    return out;
}

std::vector<QNodePtr> QubitsTopology::Choose(std::vector<qbit_t> ids) {
    std::vector<QNodePtr> out{};

    std::transform(std::begin(ids), std::end(ids), std::back_inserter(out), [&](size_t id) { return (*this)[id]; });
    return out;
}

void QubitsTopology::RemoveQubitNode(qbit_t id) {
    auto will_remove = (*this)[id];
    auto neighbour = will_remove->neighbour;
    std::accumulate(neighbour.begin(), neighbour.end(), will_remove,
                    [&](auto init, const auto near_id) { return *init < (*this)[near_id]; });
    this->qubits.erase(will_remove->id);
}

void QubitsTopology::RemoveIsolateNode() {
    VT<qbit_t> isolate_qid;
    for (auto& [id, qubit] : this->qubits) {
        if (qubit->neighbour.size() == 0) {
            isolate_qid.push_back(id);
        }
    }
    for (auto qid : isolate_qid) {
        this->RemoveQubitNode(qid);
    }
}

void QubitsTopology::IsolateWithNear(qbit_t id) {
    auto current = (*this)[id];
    auto neighbour = current->neighbour;
    std::accumulate(neighbour.begin(), neighbour.end(), current,
                    [&](auto init, const auto near_id) { return *init < (*this)[near_id]; });
}

bool QubitsTopology::IsCoupledWith(qbit_t id1, qbit_t id2) {
    auto q1 = (*this)[id1];
    auto q2 = (*this)[id2];
    return q1->neighbour.count(q2->id) > 0;
}

std::unordered_map<qbit_t, QNodePtr> QubitsTopology::Dict() {
    return this->qubits;
}

void QubitsTopology::AddQubitNode(const QNodePtr& qubit) {
    if (this->HasQubitNode(qubit->id)) {
        throw std::runtime_error("qubit with id " + std::to_string(qubit->id) + " already exists.");
    }
    this->qubits[qubit->id] = qubit;
}

bool QubitsTopology::HasQubitNode(qbit_t id) {
    return this->qubits.count(id) > 0;
}

// =============================================================================

LinearQubits::LinearQubits(qbit_t n_qubits) {
    if (n_qubits <= 0) {
        throw std::runtime_error("n_qubits should be greater than 0, but get " + std::to_string(n_qubits) + ".");
    }
    for (qbit_t i = 0; i < n_qubits; i++) {
        QNodePtr qubit = std::make_shared<QubitNode>(i, "#000000", i, 0.0);
        this->qubits.insert(std::unordered_map<qbit_t, QNodePtr>::value_type(i, qubit));
    }
    auto next_node = (*this)[0];
    for (qbit_t i = 1; i < n_qubits; i++) {
        next_node = (*next_node >> (*this)[i]);
    }
}

GridQubits::GridQubits(qbit_t n_row, qbit_t n_col) : n_row(n_row), n_col(n_col) {
    if (n_row <= 0) {
        throw std::runtime_error("n_row should be greater than 0, but get " + std::to_string(n_row) + ".");
    }
    if (n_col <= 0) {
        throw std::runtime_error("n_col should be greater than 0, but get " + std::to_string(n_col) + ".");
    }
    for (qbit_t r = 0; r < n_row; r++) {
        for (qbit_t c = 0; c < n_col; c++) {
            qbit_t id = r * n_col + c;
            QNodePtr qubit = std::make_shared<QubitNode>(id, "#000000", c, r);
            this->qubits.insert(std::unordered_map<qbit_t, QNodePtr>::value_type(id, qubit));
        }
    }
    for (qbit_t r = 0; r < n_row; r++) {
        auto next_node = (*this)[r * n_col];
        for (qbit_t c = 1; c < n_col; c++) {
            next_node = *next_node >> (*this)[next_node->id + 1];
        }
    }
    for (qbit_t c = 0; c < n_col; c++) {
        auto next_node = (*this)[c];
        for (qbit_t r = 1; r < n_row; r++) {
            next_node = *next_node >> (*this)[next_node->id + n_col];
        }
    }
}

int GridQubits::NRow() {
    return this->n_row;
}

int GridQubits::NCol() {
    return this->n_col;
}
}  // namespace mindquantum::mapping
