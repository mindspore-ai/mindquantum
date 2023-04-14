//   Copyright 2023 <Huawei Technologies Co., Ltd>
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

#include "math/tensor/csr_matrix.hpp"

#include <cstdlib>

namespace tensor {
CsrMatrix::CsrMatrix(CsrMatrix&& t) {
    this->n_row = t.n_row;
    this->n_col = t.n_col;
    this->nnz = t.nnz;
    this->indptr_ = t.indptr_;
    this->indices_ = t.indices_;
    this->data_ = std::move(t.data_);
    t.indptr_ = nullptr;
    t.indices_ = nullptr;
    t.data_ = Tensor();
}
CsrMatrix& CsrMatrix::operator=(CsrMatrix&& t) {
    this->n_row = t.n_row;
    this->n_col = t.n_col;
    this->nnz = t.nnz;
    if (this->indices_ != nullptr) {
        free(this->indices_);
    }
    if (this->indptr_ != nullptr) {
        free(this->indptr_);
    }
    this->indptr_ = t.indptr_;
    this->indices_ = t.indices_;
    this->data_ = std::move(t.data_);
    t.indptr_ = nullptr;
    t.indices_ = nullptr;
    t.data_ = Tensor();
    return *this;
}
CsrMatrix::CsrMatrix(const CsrMatrix& t) {
    this->n_row = t.n_row;
    this->n_col = t.n_col;
    this->nnz = t.nnz;
    this->indices_ = reinterpret_cast<size_t*>(malloc(sizeof(size_t) * nnz));
    this->indptr_ = reinterpret_cast<size_t*>(malloc(sizeof(size_t) * (n_row + 1)));
    std::memcpy(this->indices_, t.indices_, sizeof(size_t) * nnz);
    std::memcpy(this->indptr_, t.indptr_, sizeof(size_t) * (n_row + 1));
    this->data_ = t.data_;
}
CsrMatrix& CsrMatrix::operator=(const CsrMatrix& t) {
    this->n_row = t.n_row;
    this->n_col = t.n_col;
    this->nnz = t.nnz;
    if (this->indices_ != nullptr) {
        free(this->indices_);
    }
    if (this->indptr_ != nullptr) {
        free(this->indptr_);
    }
    this->indices_ = reinterpret_cast<size_t*>(malloc(sizeof(size_t) * nnz));
    this->indptr_ = reinterpret_cast<size_t*>(malloc(sizeof(size_t) * (n_row + 1)));
    std::memcpy(this->indices_, t.indices_, sizeof(size_t) * nnz);
    std::memcpy(this->indptr_, t.indptr_, sizeof(size_t) * (n_row + 1));
    this->data_ = t.data_;
    return *this;
}
CsrMatrix::CsrMatrix(size_t n_row, size_t n_col, size_t nnz, size_t* indptr, size_t* indices, Tensor&& data)
    : n_row(n_row), n_col(n_col), nnz(nnz), indptr_(indptr), indices_(indices) {
    std::swap(data_, data);
}
CsrMatrix::~CsrMatrix() {
    if (indices_ != nullptr) {
        free(indices_);
        indices_ = nullptr;
    }
    if (indptr_ != nullptr) {
        free(indptr_);
        indptr_ = nullptr;
    }
}
TDtype CsrMatrix::GetDtype() const {
    return data_.dtype;
}
TDevice CsrMatrix::GetDevice() const {
    return data_.device;
}
void CsrMatrix::CastTo(TDtype dtype) {
    if (this->GetDtype() != dtype) {
        this->data_ = this->data_.astype(dtype);
    }
}
}  // namespace tensor
