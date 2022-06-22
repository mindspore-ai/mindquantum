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

#include "python/cengines/engine_list.hpp"

#include <algorithm>
#include <iostream>
#include <string_view>

#include <pybind11/pybind11.h>

#include "python/cengines/instruction_filter.hpp"
#include "python/cengines/mapping.hpp"
#include "python/cengines/optimisation.hpp"
#include "python/cengines/printer.hpp"
#include "python/cengines/resource_counter.hpp"
#include "python/cengines/tag_remover.hpp"

namespace py = pybind11;

// =============================================================================

template <typename caster_t>
bool convert_engine(pybind11::handle src, mindquantum::cengines::engine_t& engine, std::string_view type_name) {
    if (caster_t caster; caster.load(src, true)) {
        engine = caster;
        return true;
    } else {
        std::cerr << "Failed to convert " << type_name << std::endl;
        return false;
    }
}

// =============================================================================

bool mindquantum::details::load_cengine(pybind11::handle src, cengines::engine_t& engine) {
    using py::detail::make_caster;

    auto ptype = Py_TYPE(src.ptr());
    std::string_view type_name(ptype->tp_name);

    if (type_name == "LocalOptimizer") {
        using caster_t = make_caster<python::cpp::LocalOptimizer>;
        return convert_engine<caster_t>(src, engine, type_name);
    } else if (type_name == "LinearMapper") {
        using caster_t = make_caster<python::cpp::LinearMapper>;
        return convert_engine<caster_t>(src, engine, type_name);
    } else if (type_name == "GridMapper") {
        using caster_t = make_caster<python::cpp::GridMapper>;
        return convert_engine<caster_t>(src, engine, type_name);
    } else if (type_name == "projectq.cengines._mindquantum_cxx_cengines.CppGraphMapper") {
        using caster_t = make_caster<python::CppGraphMapper>;
        return convert_engine<caster_t>(src, engine, type_name);
    } else if (type_name == "CommandPrinter") {
        using language_t = cengines::CppPrinter::language_t;
        engine = cengines::CppPrinter(language_t::projectq);
        return true;
    } else if (type_name == "projectq.cengines._mindquantum_cxx_cengines.CppPrinter") {
        using caster_t = make_caster<python::CommandPrinter>;
        return convert_engine<caster_t>(src, engine, type_name);
    } else if (type_name == "ResourceCounter") {
        using caster_t = make_caster<python::ResourceCounter>;
        return convert_engine<caster_t>(src, engine, type_name);
    } else if (type_name == "TagRemover") {
        using caster_t = make_caster<python::cpp::TagRemover>;
        return convert_engine<caster_t>(src, engine, type_name);
    }
    // TODO(dnguyen): Check that name is correct (maybe add _mindquantum_cxx_cengines)
    else if (type_name == "InstructionFilter"
             || type_name == "projectq.cengines._replacer._replacer.InstructionFilter") {
        using caster_t = make_caster<python::cpp::InstructionFilter>;
        return convert_engine<caster_t>(src, engine, type_name);
    } else {
        std::cerr << "Unsupported engine type: " << type_name << std::endl;
    }

    return false;
}
