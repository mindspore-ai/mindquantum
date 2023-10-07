/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef MQ_CONFIG_TSL_ORDERED_MAP_HPP
#define MQ_CONFIG_TSL_ORDERED_MAP_HPP

#include <algorithm>
#include <type_traits>
#include <utility>

#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <tsl/ordered_map.h>

#include "config/config.h"

namespace tsl {
template <typename iterator_t, typename function_t>
MQ_NODISCARD constexpr function_t for_each(iterator_t first, const iterator_t& last, function_t func) {
    if constexpr (std::is_const_v<std::remove_pointer_t<typename std::iterator_traits<iterator_t>::pointer>>) {
        return std::for_each(first, last, std::move(func));
    } else {
        for (; first != last; ++first) {
            func(first.value());
        }
        return func;
    }
}

template <typename iterator_t, typename predicate_t>
MQ_NODISCARD constexpr iterator_t remove_if(iterator_t first, iterator_t last, predicate_t pred) {
    first = std::find_if(first, last, pred);
    if (first != last) {
        auto it = first;
        while (++it != last) {
            if (!pred(*it)) {
                first.value() = std::move(it.value());
                ++first;
            }
        }
    }
    return first;
}
}  // namespace tsl

// =============================================================================

namespace nlohmann {
template <typename key_t, typename value_t, typename hash_t>
struct adl_serializer<tsl::ordered_map<key_t, value_t, hash_t>> {
    using type = tsl::ordered_map<key_t, value_t, hash_t>;

    static void to_json(json& json_data, const type& ordered_map) {
        if (std::empty(ordered_map)) {
            json_data = nlohmann::json::array();
        } else {
            for (const auto& key_value : ordered_map) {
                json_data.push_back(key_value);
            }
        }
    }

    static void from_json(const json& json_data, type& ordered_map) {
        if (!json_data.is_array()) {
            throw detail::type_error::create(302, detail::concat("type must be array, but is ", json_data.type_name()),
                                             &json_data);
        }
        ordered_map.clear();
        for (const auto& param : json_data) {
            if (!param.is_array()) {
                throw detail::type_error::create(302, detail::concat("type must be array, but is ", param.type_name()),
                                                 &json_data);
            }
            ordered_map.emplace(param.at(0).template get<key_t>(), param.at(1).template get<value_t>());
        }
    }
};
}  // namespace nlohmann

#endif /* MQ_CONFIG_TSL_ORDERED_MAP_HPP */
