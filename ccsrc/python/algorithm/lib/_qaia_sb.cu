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

#include <memory>

#include <cuda_fp16.h>

#include <fmt/format.h>
#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "algorithm/qaia/csr_base.h"
#include "algorithm/qaia/detail/gpu_sb.cuh"
#include "algorithm/qaia/detail/para.h"
#include "algorithm/qaia/detail/sb_helper.h"
#include "config/constexpr_type_name.h"
#include "config/format/std_complex.h"
#include "config/type_traits.h"
#include "core/mq_base_types.h"

namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)
using mindquantum::Index;

template <int SB, typename T, bool H>
void sb_update(py::object& csr, py::array_t<double>& x, py::array_t<double>& h, int B, float xi, float delta, float dt,
               int n_iter) {
    auto indices = csr.attr("indices").cast<py::array_t<Index>>();
    auto indptr = csr.attr("indptr").cast<py::array_t<Index>>();
    auto data = csr.attr("data").cast<py::array_t<double>>();
    int nnz = data.size();
    Index* raw_indices = static_cast<Index*>(indices.request().ptr);
    Index* raw_indptr = static_cast<Index*>(indptr.request().ptr);
    double* raw_data = static_cast<double*>(data.request().ptr);
    auto shape = csr.attr("shape").cast<py::tuple>();
    int nrows = shape[0].cast<int>();
    int ncols = shape[1].cast<int>();
    int N = nrows;
    mindquantum::sparse::CsrBase<double> csr_matrix(N, nnz, raw_indptr, raw_indices, raw_data);

    double* raw_x = static_cast<double*>(x.request().ptr);
    mindquantum::algorithm::qaia::detail::Para paras(B, xi, delta, dt, n_iter);

    double* raw_h = static_cast<double*>(h.request().ptr);
    int ndim = h.ndim();

    mindquantum::algorithm::qaia::detail::SBUpdater<SB, T, H>::update(csr_matrix, raw_x, paras, raw_h, ndim);
}

PYBIND11_MODULE(_qaia_sb, module) {
    module.def("cuda_init", mindquantum::algorithm::qaia::detail::SBBase::cublas_warmup, "warmup cuBLAS");
    module.def("bsb_update_h_int8", &sb_update<0, int8_t, true>, "BSB update func(int8_t) with h");
    module.def("bsb_update_h_half", &sb_update<0, half, true>, "BSB update func(half) with h");
    module.def("dsb_update_h_int8", &sb_update<1, int8_t, true>, "DSB update func(int8_t) with h");
    module.def("dsb_update_h_half", &sb_update<1, half, true>, "DSB update func(half) with h");

    module.def("bsb_update_int8", &sb_update<0, int8_t, false>, "BSB update func(int8_t) without h");
    module.def("bsb_update_half", &sb_update<0, half, false>, "BSB update func(half) without h");
    module.def("dsb_update_int8", &sb_update<1, int8_t, false>, "DSB update func(int8_t) without h");
    module.def("dsb_update_half", &sb_update<1, half, false>, "DSB update func(half) without h");
}