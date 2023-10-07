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

#ifndef MATH_OPERATORS_UTILS
#define MATH_OPERATORS_UTILS
#include <cstdint>
#include <list>
#include <map>
#include <utility>
#include <vector>

#include "math/pr/parameter_resolver.h"
namespace operators {
using key_t = std::vector<uint64_t>;
using value_t = parameter::ParameterResolver;
using compress_term_t = std::pair<key_t, value_t>;

struct KeyCompare {
    bool operator()(const key_t& a, const key_t& b) const;
};

// -----------------------------------------------------------------------------

class QTerm_t {
 public:
    void insert(const key_t& key, const value_t& value) {
        if (m_map.find(key) != m_map.end()) {
            m_list.erase(m_map[key]);
            m_map.erase(key);
        }
        m_list.push_back({key, value});
        m_map[key] = --m_list.end();
    }
    void erase(const key_t& key) {
        if (m_map.find(key) != m_map.end()) {
            m_list.erase(m_map[key]);
            m_map.erase(key);
        }
    }
    void insert(const compress_term_t& t) {
        this->insert(t.first, t.second);
    }
    value_t& operator[](const key_t& key) {
        return (*m_map[key]).second;
    }

    typename std::list<std::pair<key_t, value_t>>::iterator begin() {
        return m_list.begin();
    }

    typename std::list<std::pair<key_t, value_t>>::iterator end() {
        return m_list.end();
    }

    typename std::list<std::pair<key_t, value_t>>::const_iterator begin() const {
        return m_list.begin();
    }

    typename std::list<std::pair<key_t, value_t>>::const_iterator end() const {
        return m_list.end();
    }
    size_t size() const {
        return this->m_map.size();
    }

    // -----------------------------------------------------------------------------

    QTerm_t() = default;
    QTerm_t(const QTerm_t& other) {
        for (auto& p : other.m_list) {
            insert(p);
        }
    }

    QTerm_t(QTerm_t&& other) {
        m_list = std::move(other.m_list);
        m_map = std::move(other.m_map);
    }
    QTerm_t& operator=(const QTerm_t& other) {
        if (this != &other) {
            m_list.clear();
            m_map.clear();
            for (auto& p : other.m_list) {
                insert(p);
            }
        }
        return *this;
    }
    QTerm_t& operator=(QTerm_t&& other) {
        if (this != &other) {
            m_list = std::move(other.m_list);
            m_map = std::move(other.m_map);
        }
        return *this;
    }

 public:
    std::list<compress_term_t> m_list;
    std::map<key_t, std::list<compress_term_t>::iterator, KeyCompare> m_map;
};
}  // namespace operators
#endif /* MATH_OPERATORS_UTILS */
