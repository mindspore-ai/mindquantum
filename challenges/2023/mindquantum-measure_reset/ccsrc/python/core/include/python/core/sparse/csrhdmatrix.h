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

#ifndef MQ_PYTHON_CSRHDMATRIX_HPP
#define MQ_PYTHON_CSRHDMATRIX_HPP

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "core/sparse/csrhdmatrix.h"

namespace mindquantum::python {
template <typename T>
struct CsrHdMatrix : sparse::CsrHdMatrix<T> {
    using sparse::CsrHdMatrix<T>::CsrHdMatrix;

    using base_t = sparse::CsrHdMatrix<T>;
    using base_t::data_;
    using base_t::dim_;
    using base_t::indices_;
    using base_t::indptr_;
    using base_t::nnz_;

    CsrHdMatrix(Index dim, Index nnz, pybind11::array_t<Index> indptr, pybind11::array_t<Index> indices,
                pybind11::array_t<CT<T>> data)
        : base_t(dim, nnz, reinterpret_cast<Index *>(malloc(indptr.size() * sizeof(Index))),
                 reinterpret_cast<Index *>(malloc(indices.size() * sizeof(Index))),
                 (CTP<T>) malloc(data.size() * sizeof(CT<T>))) {
        Index *indptr_py = static_cast<Index *>(indptr.request().ptr);
        Index *indices_py = static_cast<Index *>(indices.request().ptr);
        CTP<T> data_py = static_cast<CT<T> *>(data.request().ptr);
        for (size_t i = 0; i < static_cast<size_t>(data.size()); i++) {
            indices_[i] = indices_py[i];
            data_[i] = data_py[i];
        }
        for (size_t i = 0; i < static_cast<size_t>(indptr.size()); i++) {
            indptr_[i] = indptr_py[i];
        }
    }
};
}  // namespace mindquantum::python

#endif /* MQ_PYTHON_CSRHDMATRIX_HPP */
