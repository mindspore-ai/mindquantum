/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_SYMBOL_HPP
#define INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_SYMBOL_HPP

// Name of a symbol exported by the OpenBLAS build mqmps links against.
//
// BLAS_SYMBOL_PREFIX follows the scipy-openblas convention (`scipy_` there, empty for a stock OpenBLAS). It is supplied
// by the OpenBLAS::OpenBLAS CMake target, which only defines it when the prefix is non-empty.
#ifndef BLAS_SYMBOL_PREFIX
#    define BLAS_SYMBOL_PREFIX
#endif

#define MQ_LINALG_SYM_CAT_IMPL(prefix, name) prefix##name
#define MQ_LINALG_SYM_CAT(prefix, name)      MQ_LINALG_SYM_CAT_IMPL(prefix, name)
#define MQ_LINALG_SYM(name)                  MQ_LINALG_SYM_CAT(BLAS_SYMBOL_PREFIX, name)

#endif  // INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_SYMBOL_HPP
