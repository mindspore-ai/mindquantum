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

#ifndef TERMS_COEFF_DICT_HPP
#define TERMS_COEFF_DICT_HPP

#include <utility>

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/indexed_by.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/tag.hpp>
#include <boost/multi_index_container.hpp>

#include <nlohmann/json.hpp>

#include "ops/gates/term_value.hpp"

namespace mindquantum::ops {
namespace order {
struct insertion {};
struct hashed {};
}  // namespace order

namespace impl_ {
template <typename coeff_t>
using value_t = std::pair<terms_t, coeff_t>;
}  // namespace impl_

template <typename coeff_t>
using term_dict_t = boost::multi_index_container<
    impl_::value_t<coeff_t>,
    boost::multi_index::indexed_by<
        boost::multi_index::random_access<boost::multi_index::tag<order::insertion>>,  // insertion order
        boost::multi_index::hashed_unique<
            boost::multi_index::tag<order::hashed>,
            boost::multi_index::member<impl_::value_t<coeff_t>, terms_t, &impl_::value_t<coeff_t>::first>>>>;
}  // namespace mindquantum::ops

// =============================================================================

namespace nlohmann {
template <typename coeff_t>
struct adl_serializer<mindquantum::ops::term_dict_t<coeff_t>> {
    using type = mindquantum::ops::term_dict_t<coeff_t>;
    using key_t = typename type::value_type::first_type;
    using value_t = typename type::value_type::second_type;

    static void to_json(json& json_data, const type& ordered_map) {  // NOLINT(runtime/references)
        if (std::empty(ordered_map)) {
            json_data = nlohmann::json::array();
        } else {
            for (const auto& key_value : ordered_map) {
                json_data.push_back(key_value);
            }
        }
    }

    static void from_json(const json& json_data, type& ordered_map) {  // NOLINT(runtime/references)
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
            ordered_map.emplace_back(param.at(0).template get<key_t>(), param.at(1).template get<value_t>());
        }
    }
};
}  // namespace nlohmann

#endif /* TERMS_COEFF_DICT_HPP */
