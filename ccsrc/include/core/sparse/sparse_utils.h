/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#ifndef MINDQUANTUM_SPARSE_SPARSE_UTILS_H_
#define MINDQUANTUM_SPARSE_SPARSE_UTILS_H_

#include <complex>
#include <vector>

#include "core/utils.h"

namespace mindquantum {
namespace sparse {
template <typename T>
void csr_plus_csr(Index dim, const Index *a_indptr, const Index *aj, const T *ad, const Index *b_indptr,
                  const Index *bj, const T *bd, Index *cp, Index *cj, T *cd) {
    cp[0] = 0;
    Index nnz = 0;
    for (Index i = 0; i < dim; i++) {
        Index ap = a_indptr[i];
        Index bp = b_indptr[i];
        Index a_end = a_indptr[i + 1];
        Index b_end = b_indptr[i + 1];

        while (ap < a_end && bp < b_end) {
            Index a_j = aj[ap];
            Index b_j = bj[bp];

            if (a_j == b_j) {
                T result = ad[ap] + bd[bp];
                if (std::abs(result) > PRECISION) {
                    cj[nnz] = a_j;
                    cd[nnz] = result;
                    nnz++;
                }
                ap++;
                bp++;
            } else if (a_j < b_j) {
                T result = ad[ap];
                if (std::abs(result) > PRECISION) {
                    cj[nnz] = a_j;
                    cd[nnz] = result;
                    nnz++;
                }
                ap++;
            } else {
                T result = bd[bp];
                if (std::abs(result) > PRECISION) {
                    cj[nnz] = b_j;
                    cd[nnz] = result;
                    nnz++;
                }
                bp++;
            }
        }

        while (ap < a_end) {
            T result = ad[ap];
            if (std::abs(result) > PRECISION) {
                cj[nnz] = aj[ap];
                cd[nnz] = result;
                nnz++;
            }
            ap++;
        }

        while (bp < b_end) {
            T result = bd[bp];
            if (std::abs(result) > PRECISION) {
                cj[nnz] = bj[bp];
                cd[nnz] = result;
                nnz++;
            }
            bp++;
        }

        cp[i + 1] = nnz;
    }
}
}  // namespace sparse
}  // namespace mindquantum
#endif  // MINDQUANTUM_SPARSE_SPARSE_UTILS_H_
