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

#ifndef INCLUDE_QAIA_DETAIL_GPU_SB_CUH
#define INCLUDE_QAIA_DETAIL_GPU_SB_CUH

#include "algorithm/qaia/csr_base.h"
#include "algorithm/qaia/detail/para.h"

namespace mindquantum::algorithm::qaia::detail {

struct SBBase {
    static void dSB_update_int8(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras);

    static void bSB_update_int8(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras);

    static void bSB_update_fp16(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras);

    static void dSB_update_fp16(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras);

    static void dSB_update_h_int8(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras, double* h,
                                  int h_size);

    static void bSB_update_h_int8(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras, double* h,
                                  int h_size);

    static void bSB_update_h_fp16(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras, double* h,
                                  int h_size);

    static void dSB_update_h_fp16(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras, double* h,
                                  int h_size);

    static void cublas_warmup(int N, int B);
};

}  // namespace mindquantum::algorithm::qaia::detail

#endif
