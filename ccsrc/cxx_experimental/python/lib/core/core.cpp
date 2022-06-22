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

#include "python/core/core.hpp"

#include "cengines/cpp_resource_counter.hpp"
#include "core/details/visitor.hpp"
#include "python/cengines/resource_counter.hpp"

void mindquantum::python::CppCore::flush() {
    base_t::flush();

    std::for_each(std::begin(engine_list_), std::end(engine_list_), [](const auto& engine) {
        std::visit(overload{[](const cengines::ResourceCounter& engine) {
                                const auto* resource_counter = static_cast<const python::ResourceCounter*>(&engine);
                                resource_counter->write_data_to_python();
                            },
                            [](auto&&) {}},
                   engine);
    });
}
