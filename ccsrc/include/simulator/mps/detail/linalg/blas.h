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

#ifndef INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_BLAS_HPP
#define INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_BLAS_HPP

#include <cctype>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>

#include "simulator/mps/detail/linalg/matrix.h"

namespace mindquantum::sim::mps::detail::linalg {
using blas_int = int;

static_assert(sizeof(blas_int) == 4, "mqmps BLAS/LAPACK layer requires LP64 (32-bit BLAS integers).");

namespace blas::detail {
inline blas_int ToBlasInt(size_t v, const char* what) {
    if (v > static_cast<size_t>(std::numeric_limits<blas_int>::max())) {
        throw std::runtime_error(std::string("mqmps: BLAS dimension overflow for ") + what);
    }
    return static_cast<blas_int>(v);
}
}  // namespace blas::detail

extern "C" {
void cgemm_(const char* transa, const char* transb, const blas_int* m, const blas_int* n, const blas_int* k,
            const std::complex<float>* alpha, const std::complex<float>* a, const blas_int* lda,
            const std::complex<float>* b, const blas_int* ldb, const std::complex<float>* beta,
            std::complex<float>* c, const blas_int* ldc);
void zgemm_(const char* transa, const char* transb, const blas_int* m, const blas_int* n, const blas_int* k,
            const std::complex<double>* alpha, const std::complex<double>* a, const blas_int* lda,
            const std::complex<double>* b, const blas_int* ldb, const std::complex<double>* beta,
            std::complex<double>* c, const blas_int* ldc);
}  // extern "C"

template <typename T>
struct Blas;

template <>
struct Blas<std::complex<float>> {
    static void gemm(const char* transa, const char* transb, const blas_int* m, const blas_int* n, const blas_int* k,
                     const std::complex<float>* alpha, const std::complex<float>* a, const blas_int* lda,
                     const std::complex<float>* b, const blas_int* ldb, const std::complex<float>* beta,
                     std::complex<float>* c, const blas_int* ldc) {
        cgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
};

template <>
struct Blas<std::complex<double>> {
    static void gemm(const char* transa, const char* transb, const blas_int* m, const blas_int* n, const blas_int* k,
                     const std::complex<double>* alpha, const std::complex<double>* a, const blas_int* lda,
                     const std::complex<double>* b, const blas_int* ldb, const std::complex<double>* beta,
                     std::complex<double>* c, const blas_int* ldc) {
        zgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
};

template <typename T>
inline void Gemm(char transa, char transb, T alpha, MatrixView<const T> a, MatrixView<const T> b, T beta,
                 MatrixView<T> c) {
    const auto norm_trans = [](char t, const char* what) {
        const char u = static_cast<char>(std::toupper(static_cast<unsigned char>(t)));
        if (u != 'N' && u != 'T' && u != 'C') {
            throw std::runtime_error(std::string("mqmps: invalid GEMM transpose flag for ") + what + ".");
        }
        return u;
    };
    const char t_a = norm_trans(transa, "A");
    const char t_b = norm_trans(transb, "B");

    const bool ta = (t_a != 'N');
    const bool tb = (t_b != 'N');

    const size_t m = ta ? a.cols() : a.rows();
    const size_t k_a = ta ? a.rows() : a.cols();
    const size_t k_b = tb ? b.cols() : b.rows();
    const size_t n = tb ? b.rows() : b.cols();

    if (k_a != k_b || c.rows() != m || c.cols() != n) {
        throw std::runtime_error("mqmps: GEMM dimension mismatch.");
    }

    const auto m_i = blas::detail::ToBlasInt(m, "m");
    const auto n_i = blas::detail::ToBlasInt(n, "n");
    const auto k_i = blas::detail::ToBlasInt(k_a, "k");
    const auto lda_i = blas::detail::ToBlasInt(a.leading_dim(), "lda");
    const auto ldb_i = blas::detail::ToBlasInt(b.leading_dim(), "ldb");
    const auto ldc_i = blas::detail::ToBlasInt(c.leading_dim(), "ldc");

    Blas<T>::gemm(&t_a, &t_b, &m_i, &n_i, &k_i, &alpha, a.data(), &lda_i, b.data(), &ldb_i, &beta, c.data(), &ldc_i);
}
}  // namespace mindquantum::sim::mps::detail::linalg

#endif  // INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_BLAS_HPP
