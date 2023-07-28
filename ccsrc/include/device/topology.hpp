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

#ifndef MINDQUANTUM_DEVICE_TOPOLOGY_HPP_
#define MINDQUANTUM_DEVICE_TOPOLOGY_HPP_
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mindquantum::mapping {
using qbit_t = int64_t;
template <typename T>
using VT = std::vector<T>;

// -----------------------------------------------------------------------------

struct QubitNode;
using QNodePtr = std::shared_ptr<QubitNode>;

struct QubitNode : std::enable_shared_from_this<QubitNode> {
    qbit_t id = 0;
    std::string color = "#000000";
    double poi_x = 0;
    double poi_y = 0;
    std::set<qbit_t> neighbour = {};

    // -----------------------------------------------------------------------------

    QubitNode() = default;
    explicit QubitNode(qbit_t id, std::string color = "#000000", double poi_x = 0.0, double poi_y = 0.0,
                       const std::set<qbit_t>& neighbour = {});

    // -----------------------------------------------------------------------------

    void SetPoi(double poi_x, double poi_y);

    // left_node << right_node
    // connect this two nodes, and return left_node
    QNodePtr operator<<(const QNodePtr& other);

    // left_node >> right_node
    // connect this tow nodes, and return right_node
    QNodePtr operator>>(const QNodePtr& other);

    // left_node < right_node
    // disconnect this two nodes, and return left_node
    QNodePtr operator<(const QNodePtr& other);

    // left_node > right_node
    // disconnect this two nodes, and return right_node
    QNodePtr operator>(const QNodePtr& other);

    // int id = QubitNode(0)
    // automatic convert a QubitNode to int.
    operator qbit_t();
};

// =============================================================================

class QubitsTopology {
 public:
    QubitsTopology() = default;
    explicit QubitsTopology(const VT<QNodePtr>& qubits);

    QNodePtr operator[](qbit_t id);
    int size();
    int NEdges();
    std::set<qbit_t> AllQubitID();

    void SetPosition(qbit_t id, double poi_x, double poi_y);
    void SetPosition(std::map<qbit_t, std::pair<double, double>> positions);
    void SetColor(qbit_t id, std::string color);
    void SetColor(std::map<qbit_t, std::string> colors);

    std::set<std::pair<qbit_t, qbit_t>> EdgesWithID();
    std::set<std::pair<std::pair<double, double>, std::pair<double, double>>> EdgesWithPoi();

    std::vector<QNodePtr> Choose(std::vector<qbit_t> ids);
    void RemoveQubitNode(qbit_t id);
    void AddQubitNode(const QNodePtr& qubit);
    bool HasQubitNode(qbit_t id);
    void RemoveIsolateNode();
    void IsolateWithNear(qbit_t id);
    bool IsCoupledWith(qbit_t id1, qbit_t id2);
    template <typename F>
    void ForEachQubitNode(F&& func) {
        std::for_each(qubits.begin(), qubits.end(), func);
    }
    std::unordered_map<qbit_t, QNodePtr> Dict();

 protected:
    std::unordered_map<qbit_t, QNodePtr> qubits;
};

class LinearQubits : public QubitsTopology {
 public:
    explicit LinearQubits(qbit_t n_qubits);
};

class GridQubits : public QubitsTopology {
 public:
    GridQubits(qbit_t n_row, qbit_t n_col);
    int NRow();
    int NCol();

 private:
    int n_row = 0;
    int n_col = 0;
};
}  // namespace mindquantum::mapping

#endif
