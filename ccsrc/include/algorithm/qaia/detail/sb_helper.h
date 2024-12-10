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

#ifndef SB_HELPER_H
#define SB_HELPER_H

#include <cuda_fp16.h>

#include "algorithm/qaia/csr_base.h"
#include "algorithm/qaia/detail/gpu_sb.cuh"
#include "algorithm/qaia/detail/para.h"

namespace mindquantum::algorithm::qaia::detail {

// Helper structure, using template specialization to select different update functions
template <int SB, typename T, bool H>
struct SBUpdater;

template <>
struct SBUpdater<0, half, true> {
    static void update(const mindquantum::sparse::CsrBase<double>& csr_matrix, double* raw_x,
                       const mindquantum::algorithm::qaia::detail::Para& paras, double* raw_h, int h_size) {
        mindquantum::algorithm::qaia::detail::SBBase::bSB_update_h_fp16(csr_matrix, raw_x, paras, raw_h, h_size);
    }
};

template <>
struct SBUpdater<0, int8_t, true> {
    static void update(const mindquantum::sparse::CsrBase<double>& csr_matrix, double* raw_x,
                       const mindquantum::algorithm::qaia::detail::Para& paras, double* raw_h, int h_size) {
        mindquantum::algorithm::qaia::detail::SBBase::bSB_update_h_int8(csr_matrix, raw_x, paras, raw_h, h_size);
    }
};

template <>
struct SBUpdater<1, half, true> {
    static void update(const mindquantum::sparse::CsrBase<double>& csr_matrix, double* raw_x,
                       const mindquantum::algorithm::qaia::detail::Para& paras, double* raw_h, int h_size) {
        mindquantum::algorithm::qaia::detail::SBBase::dSB_update_h_fp16(csr_matrix, raw_x, paras, raw_h, h_size);
    }
};

template <>
struct SBUpdater<1, int8_t, true> {
    static void update(const mindquantum::sparse::CsrBase<double>& csr_matrix, double* raw_x,
                       const mindquantum::algorithm::qaia::detail::Para& paras, double* raw_h, int h_size) {
        mindquantum::algorithm::qaia::detail::SBBase::dSB_update_h_int8(csr_matrix, raw_x, paras, raw_h, h_size);
    }
};

template <>
struct SBUpdater<0, half, false> {
    static void update(const mindquantum::sparse::CsrBase<double>& csr_matrix, double* raw_x,
                       const mindquantum::algorithm::qaia::detail::Para& paras, double* raw_h, int h_size) {
        mindquantum::algorithm::qaia::detail::SBBase::bSB_update_fp16(csr_matrix, raw_x, paras);
    }
};

template <>
struct SBUpdater<0, int8_t, false> {
    static void update(const mindquantum::sparse::CsrBase<double>& csr_matrix, double* raw_x,
                       const mindquantum::algorithm::qaia::detail::Para& paras, double* raw_h, int h_size) {
        mindquantum::algorithm::qaia::detail::SBBase::bSB_update_int8(csr_matrix, raw_x, paras);
    }
};

template <>
struct SBUpdater<1, half, false> {
    static void update(const mindquantum::sparse::CsrBase<double>& csr_matrix, double* raw_x,
                       const mindquantum::algorithm::qaia::detail::Para& paras, double* raw_h, int h_size) {
        mindquantum::algorithm::qaia::detail::SBBase::dSB_update_fp16(csr_matrix, raw_x, paras);
    }
};

template <>
struct SBUpdater<1, int8_t, false> {
    static void update(const mindquantum::sparse::CsrBase<double>& csr_matrix, double* raw_x,
                       const mindquantum::algorithm::qaia::detail::Para& paras, double* raw_h, int h_size) {
        mindquantum::algorithm::qaia::detail::SBBase::dSB_update_int8(csr_matrix, raw_x, paras);
    }
};

}  // namespace mindquantum::algorithm::qaia::detail
#endif
