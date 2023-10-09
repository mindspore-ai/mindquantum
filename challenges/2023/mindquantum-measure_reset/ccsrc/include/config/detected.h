/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef MQ_CONFIG_DETECTED_HPP
#define MQ_CONFIG_DETECTED_HPP

#include "config/config.h"

#ifdef HAS_STD_DETECTED_TS2
#    include <experimental/type_traits>

namespace mindquantum {
using std::experimental::detected_or;
using std::experimental::detected_t;
using std::experimental::is_detected;

using std::experimental::detected_or_t;
using std::experimental::is_detected_convertible;
using std::experimental::is_detected_convertible_v;
using std::experimental::is_detected_exact;
using std::experimental::is_detected_exact_v;
using std::experimental::is_detected_v;
}  // namespace mindquantum
#else
#    include <type_traits>

namespace mindquantum {
namespace detail {
template <class...>
using void_t = void;

struct nonesuch {
    nonesuch() = delete;
    ~nonesuch() = delete;
    nonesuch(nonesuch const&) = delete;
    void operator=(nonesuch const&) = delete;
};

template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector {
    using value_t = std::false_type;
    using type = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type = Op<Args...>;
};
}  // namespace detail

template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<detail::nonesuch, void, Op, Args...>::value_t;

template <template <class...> class Op, class... Args>
using detected_t = typename detail::detector<detail::nonesuch, void, Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using detected_or = detail::detector<Default, void, Op, Args...>;

// --------------------------------

template <class Default, template <class...> class Op, class... Args>
using detected_or_t = typename detected_or<Default, Op, Args...>::type;

template <class Expected, template <class...> class Op, class... Args>
using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

template <class Expected, template <class...> class Op, class... Args>
constexpr inline bool is_detected_exact_v = is_detected_exact<Expected, Op, Args...>::value;

template <template <class...> class Op, class... Args>
constexpr inline bool is_detected_v = is_detected<Op, Args...>::value;

template <class To, template <class...> class Op, class... Args>
using is_detected_convertible = std::is_convertible<To, detected_t<Op, Args...>>;

template <class To, template <class...> class Op, class... Args>
constexpr inline bool is_detected_convertible_v = is_detected_convertible<To, Op, Args...>::value;
}  // namespace mindquantum
#endif  // COMPILER_TYPE_TRAITS_TS2

#endif /* MQ_CONFIG_DETECTED_HPP */
