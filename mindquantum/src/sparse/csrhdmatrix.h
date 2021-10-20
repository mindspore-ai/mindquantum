/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDQUANTUM_SPARSE_CSR_HD_MATRIX_H_
#define MINDQUANTUM_SPARSE_CSR_HD_MATRIX_H_
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "core/utils.h"

namespace mindquantum {
namespace sparse {
namespace py = pybind11;
template <typename T>
struct CsrHdMatrix {
    Index dim_;
    Index nnz_;
    Index *indptr_;
    Index *indices_;
    CTP<T> data_;

    void FreeMemory() {
        if (indptr_ != nullptr) {
            free(indptr_);
        }
        if (indices_ != nullptr) {
            free(indices_);
        }
        if (data_ != nullptr) {
            free(data_);
        }
        indptr_ = nullptr;
        indices_ = nullptr;
        data_ = nullptr;
    }
    void Reset() {
        FreeMemory();
        indptr_ = nullptr;
        indices_ = nullptr;
        data_ = nullptr;
    }
    ~CsrHdMatrix() {
        FreeMemory();
    }
    CsrHdMatrix() : dim_(0), nnz_(0), indptr_(nullptr), indices_(nullptr), data_(nullptr) {
    }
    CsrHdMatrix(Index dim, Index nnz, Index *indptr, Index *indices, CTP<T> data)
        : dim_(dim), nnz_(nnz), indptr_(indptr), indices_(indices), data_(data) {
    }
    CsrHdMatrix(Index dim, Index nnz, py::array_t<Index> indptr, py::array_t<Index> indices, py::array_t<CT<T>> data)
        : dim_(dim), nnz_(nnz) {
        indptr_ = reinterpret_cast<Index *>(malloc(indptr.size() * sizeof(Index)));
        indices_ = reinterpret_cast<Index *>(malloc(indices.size() * sizeof(Index)));
        data_ = (CTP<T>) malloc(data.size() * sizeof(CT<T>));
        Index *indptr_py = static_cast<Index *>(indptr.request().ptr);
        Index *indices_py = static_cast<Index *>(indices.request().ptr);
        CTP<T> data_py = static_cast<CT<T> *>(data.request().ptr);
        for (size_t i = 0; i < data.size(); i++) {
            indices_[i] = indices_py[i];
            data_[i] = data_py[i];
        }
        for (size_t i = 0; i < indptr.size(); i++) {
            indptr_[i] = indptr_py[i];
        }
    }
    void PrintInfo() {
        std::cout << "<--Csr Half Diag Matrix with Dimension: ";
        std::cout << dim_ << " X " << dim_ << ", and nnz: " << nnz_ << std::endl;
        std::cout << "   Data:\n   ";
        for (Index i = 0; i < nnz_; ++i) {
            std::cout << data_[i];
            if (i != nnz_ - 1) {
                std::cout << ",";
            }
        }

        std::cout << "\n   indptr:\n   ";
        for (Index i = 0; i < dim_ + 1; i++) {
            std::cout << indptr_[i];
            if (i != dim_) {
                std::cout << ",";
            }
        }

        std::cout << "\n   indices:\n   ";
        for (Index i = 0; i < nnz_; i++) {
            std::cout << indices_[i];
            if (i != nnz_ - 1) {
                std::cout << ",";
            }
        }

        std::cout << "-->\n\n";
    }
};
}  // namespace sparse
}  // namespace mindquantum
#endif  // MINDQUANTUM_SPARSE_CSR_HD_MATRIX_H_
