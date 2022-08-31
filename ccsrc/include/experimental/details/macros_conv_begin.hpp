//   Copyright 2020 <Huawei Technologies Co., Ltd>
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

#include <type_traits>

namespace mindquantum::details {
template <typename T>
using add_cvref = std::add_lvalue_reference_t<std::add_const_t<T>>;
}  // namespace mindquantum::details

#ifndef DECLARE_ATTRIBUTE
#    define DECLARE_ATTRIBUTE(type, name)                                                                              \
     public:                                                                                                           \
        DECLARE_GETTER_SETTER(type, name)                                                                              \
     private:                                                                                                          \
        type name##_
#endif  // !DECLARE_ATTRIBUTE

#ifndef DECLARE_GETTER_SETTER
#    define DECLARE_GETTER_SETTER(type, name)                                                                          \
        void set_##name(mindquantum::details::add_cvref<type> value) {                                                 \
            name##_ = value;                                                                                           \
        }                                                                                                              \
        auto get_##name() const {                                                                                      \
            return name##_;                                                                                            \
        }
#endif  // !DECLARE_GETTER_SETTER
