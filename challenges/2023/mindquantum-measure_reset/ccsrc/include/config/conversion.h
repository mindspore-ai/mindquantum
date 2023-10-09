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

#ifndef MQ_CONFIG_CONVERSION_HPP
#define MQ_CONFIG_CONVERSION_HPP

namespace mindquantum::traits {
//! Helper class to convert some variable into some pre-defined type if required
template <typename type_t>
struct conversion_helper {
    //! No-op overload in case no conversion is required
    static auto apply(const type_t& value) {
        return value;
    }
    //! Overload called if some conversion is required.
    /*!
     * The conversion is performed using \c static_cast.
     */
    template <typename other_t>
    static auto apply(const other_t& value) {
        return static_cast<type_t>(value);
    }
};
}  // namespace mindquantum::traits

#endif /* MQ_CONFIG_CONVERSION_HPP */
