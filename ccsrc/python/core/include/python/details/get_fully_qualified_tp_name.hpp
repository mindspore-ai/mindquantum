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

#ifndef GET_FULLY_QUALIFIED_TP_NAME_HPP
#define GET_FULLY_QUALIFIED_TP_NAME_HPP

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace mindquantum::python {
inline auto get_fully_qualified_tp_name(const pybind11::handle& src) {
    return pybind11::detail::get_fully_qualified_tp_name(Py_TYPE(src.ptr()));
}
}  // namespace mindquantum::python

#endif /* GET_FULLY_QUALIFIED_TP_NAME_HPP */
