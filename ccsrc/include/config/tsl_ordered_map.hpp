//   Copyright 2022 <Huawei Technologies Co., Ltd>
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

#ifndef MQ_CONFIG_TSL_ORDERED_MAP_HPP
#define MQ_CONFIG_TSL_ORDERED_MAP_HPP

#include <nlohmann/detail/exceptions.hpp>
#include <nlohmann/json.hpp>
#include <tsl/ordered_map.h>

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
