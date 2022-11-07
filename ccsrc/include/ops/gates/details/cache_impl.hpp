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

#ifndef OPS_GATES_DETAILS_CACHE_IMPL_HPP
#define OPS_GATES_DETAILS_CACHE_IMPL_HPP

#include <limits>
#include <tuple>
#include <unordered_map>

#include <lru_cache/lru_cache.h>

#include "ops/gates/types.hpp"

namespace hash_tuple {
// Code from boost
// Reciprocal of the golden ratio helps spread entropy
//     and handles duplicates.
// See Mike Seymour in magic-numbers-in-boosthash-combine:
//     http://stackoverflow.com/questions/4948780

template <class T>
inline void hash_combine(std::size_t& seed, T const& v) {
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);  // NOLINT
}

// Recursive template code derived from Matthieu M.
template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl {
    static void apply(size_t& seed, Tuple const& tuple) {
        HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
        hash_combine(seed, std::get<Index>(tuple));
    }
};

template <class Tuple>
struct HashValueImpl<Tuple, 0> {
    static void apply(size_t& seed, Tuple const& tuple) {
        hash_combine(seed, std::get<0>(tuple));
    }
};

template <typename tuple_t>
struct hash {
    size_t operator()(tuple_t const& value) const {
        return std::hash<tuple_t>()(value);
    }
};
template <typename... types_t>
struct hash<std::tuple<types_t...>> {
    size_t operator()(std::tuple<types_t...> const& value) const {
        size_t seed = 0;
        HashValueImpl<std::tuple<types_t...>>::apply(seed, value);
        return seed;
    }
};
}  // namespace hash_tuple

namespace lru_cache {
using mindquantum::ops::types::csr_matrix_t;

template <typename key_t, typename value_t, std::size_t cache_size, bool by_access_order>
struct StaticLruCacheOptionsBase {
    using index_t = internal::index_type_for<cache_size>;

    static constexpr index_t MAX_SIZE = std::numeric_limits<index_t>::max() - 1;
    static_assert(cache_size <= MAX_SIZE);

    using Key = key_t;
    using Value = value_t;
    using IndexType = index_t;

    using Map = std::unordered_map<key_t, index_t, hash_tuple::hash<key_t>>;

    using NodeContainer = ArrayNodeContainer<internal::Node<key_t, value_t, index_t>, cache_size>;

    static constexpr bool ByAccessOrder = by_access_order;
};

template <typename coeff_t, std::size_t cache_size, bool by_access_order>
struct StaticLruCacheOptions<std::tuple<std::size_t, bool, std::size_t>, csr_matrix_t<coeff_t>, cache_size,
                             by_access_order>
    : public StaticLruCacheOptionsBase<std::tuple<std::size_t, bool, std::size_t>, csr_matrix_t<coeff_t>, cache_size,
                                       by_access_order> {};
template <typename coeff_t, std::size_t cache_size, bool by_access_order>
struct StaticLruCacheOptions<std::tuple<std::size_t, bool, std::size_t, bool, std::size_t>, csr_matrix_t<coeff_t>,
                             cache_size, by_access_order>
    : public StaticLruCacheOptionsBase<std::tuple<std::size_t, bool, std::size_t, bool, std::size_t>,
                                       csr_matrix_t<coeff_t>, cache_size, by_access_order> {};
}  // namespace lru_cache
#endif /* OPS_GATES_DETAILS_CACHE_IMPL_HPP */
