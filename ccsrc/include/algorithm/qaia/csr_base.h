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

#ifndef CSR_BASE_H
#define CSR_BASE_H

#include "core/mq_base_types.h"

namespace mindquantum::sparse {
template <typename T>
struct CsrBase {
    int dim_;
    int nnz_;
    Index *indptr_;
    Index *indices_;
    T *data_;

    CsrBase() : dim_(0), nnz_(0), indptr_(nullptr), indices_(nullptr), data_(nullptr) {
    }

    CsrBase(int dim, int nnz, Index *indptr, Index *indices, T *data)
        : dim_(dim), nnz_(nnz), indptr_(indptr), indices_(indices), data_(data) {
    }
};
}  // namespace mindquantum::sparse

#endif
