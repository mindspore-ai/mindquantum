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

#ifndef INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_LAPACK_SVD_HPP
#define INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_LAPACK_SVD_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "simulator/mps/detail/linalg/blas.h"
#include "simulator/mps/detail/linalg/matrix.h"

namespace mindquantum::sim::mps::detail::linalg {
template <typename T>
struct RealType;

template <>
struct RealType<std::complex<float>> {
    using type = float;
};

template <>
struct RealType<std::complex<double>> {
    using type = double;
};

template <typename T>
using real_type_t = typename RealType<T>::type;

extern "C" {
void cgesdd_(const char* jobz, const blas_int* m, const blas_int* n, std::complex<float>* a, const blas_int* lda,
             float* s, std::complex<float>* u, const blas_int* ldu, std::complex<float>* vt, const blas_int* ldvt,
             std::complex<float>* work, const blas_int* lwork, float* rwork, blas_int* iwork, blas_int* info);
void zgesdd_(const char* jobz, const blas_int* m, const blas_int* n, std::complex<double>* a, const blas_int* lda,
             double* s, std::complex<double>* u, const blas_int* ldu, std::complex<double>* vt, const blas_int* ldvt,
             std::complex<double>* work, const blas_int* lwork, double* rwork, blas_int* iwork, blas_int* info);
}  // extern "C"

template <typename T>
struct Lapack;

template <>
struct Lapack<std::complex<float>> {
    static void gesdd(const char* jobz, const blas_int* m, const blas_int* n, std::complex<float>* a, const blas_int* lda,
                      float* s, std::complex<float>* u, const blas_int* ldu, std::complex<float>* vt,
                      const blas_int* ldvt, std::complex<float>* work, const blas_int* lwork, float* rwork,
                      blas_int* iwork, blas_int* info) {
        cgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
    }
};

template <>
struct Lapack<std::complex<double>> {
    static void gesdd(const char* jobz, const blas_int* m, const blas_int* n, std::complex<double>* a,
                      const blas_int* lda, double* s, std::complex<double>* u, const blas_int* ldu,
                      std::complex<double>* vt, const blas_int* ldvt, std::complex<double>* work, const blas_int* lwork,
                      double* rwork, blas_int* iwork, blas_int* info) {
        zgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
    }
};

template <typename T>
struct SvdWorkspace {
    using real_t = real_type_t<T>;

    Matrix<T> U;
    Matrix<T> Vh;
    std::vector<real_t> S;

    std::vector<T> work;
    std::vector<real_t> rwork;
    std::vector<blas_int> iwork;

    // Cached dimensions for the last LAPACK LWORK query.
    // Note: LWORK is not safely cacheable by per-dimension maxima; it depends on the (m, n) pair.
    size_t m_last = 0;
    size_t n_last = 0;
};

namespace lapack::detail {
inline size_t CheckedMul(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::runtime_error("mqmps: size overflow in LAPACK workspace.");
    }
    return a * b;
}

inline blas_int ToBlasInt(size_t v, const char* what) {
    return blas::detail::ToBlasInt(v, what);
}

inline std::runtime_error SvdError(const char* routine, char jobz, size_t m, size_t n, blas_int info) {
    std::string msg = std::string("mqmps: ") + routine + " failed (jobz=" + jobz + ", m=" + std::to_string(m)
                      + ", n=" + std::to_string(n) + ", info=" + std::to_string(info) + ")";
    if (info < 0) {
        msg += "; invalid argument " + std::to_string(-info);
    } else if (info > 0) {
        msg += "; algorithm did not converge";
    }
    return std::runtime_error(msg);
}
}  // namespace lapack::detail

template <typename T>
inline void ComputeThinSvd(Matrix<T>* a, SvdWorkspace<T>* ws) {
    if (a == nullptr || ws == nullptr) {
        throw std::runtime_error("mqmps: null pointer passed to ComputeThinSvd.");
    }
    const size_t m = a->rows();
    const size_t n = a->cols();
    if (m == 0 || n == 0) {
        throw std::runtime_error("mqmps: SVD called with empty matrix.");
    }
    const size_t r = std::min(m, n);

    ws->U.resize(m, r);
    ws->Vh.resize(r, n);
    ws->S.resize(r);

    ws->iwork.resize(std::max<size_t>(ws->iwork.size(), lapack::detail::CheckedMul(8, r)));

    // For complex *gesdd with jobz='S':
    //   LRWORK >= 5*r*r + 7*r,  where r=min(m,n).
    const size_t rwork_req = lapack::detail::CheckedMul(5, lapack::detail::CheckedMul(r, r)) + lapack::detail::CheckedMul(7, r);
    ws->rwork.resize(std::max(ws->rwork.size(), rwork_req));

    const char jobz = 'S';

    // Query work size whenever the shape changes.
    // `*gesdd`'s optimal LWORK depends on the (m, n) pair (not on per-dimension maxima), so caching by
    // "largest seen m/n" can under-allocate for later calls with different aspect ratios.
    if (m != ws->m_last || n != ws->n_last) {
        T work_query{};
        blas_int lwork = -1;
        blas_int info = 0;

        const auto m_i = lapack::detail::ToBlasInt(m, "m");
        const auto n_i = lapack::detail::ToBlasInt(n, "n");
        const auto lda_i = lapack::detail::ToBlasInt(a->leading_dim(), "lda");
        const auto ldu_i = lapack::detail::ToBlasInt(ws->U.leading_dim(), "ldu");
        const auto ldvt_i = lapack::detail::ToBlasInt(ws->Vh.leading_dim(), "ldvt");

        Lapack<T>::gesdd(&jobz, &m_i, &n_i, a->data(), &lda_i, ws->S.data(), ws->U.data(), &ldu_i, ws->Vh.data(),
                         &ldvt_i, &work_query, &lwork, ws->rwork.data(), ws->iwork.data(), &info);

        if (info != 0) {
            throw lapack::detail::SvdError("gesdd(work_query)", jobz, m, n, info);
        }

        const auto lwork_opt_real = std::real(work_query);
        if (!(lwork_opt_real > 0)) {
            throw std::runtime_error("mqmps: invalid LAPACK work query result.");
        }
        const size_t lwork_req = static_cast<size_t>(std::ceil(lwork_opt_real));
        ws->work.resize(std::max(ws->work.size(), lwork_req));
        ws->m_last = m;
        ws->n_last = n;
    }

    if (ws->work.empty()) {
        ws->work.resize(1);
    }

    const auto m_i = lapack::detail::ToBlasInt(m, "m");
    const auto n_i = lapack::detail::ToBlasInt(n, "n");
    const auto lda_i = lapack::detail::ToBlasInt(a->leading_dim(), "lda");
    const auto ldu_i = lapack::detail::ToBlasInt(ws->U.leading_dim(), "ldu");
    const auto ldvt_i = lapack::detail::ToBlasInt(ws->Vh.leading_dim(), "ldvt");
    const auto lwork_i = lapack::detail::ToBlasInt(ws->work.size(), "lwork");

    blas_int info = 0;
    Lapack<T>::gesdd(&jobz, &m_i, &n_i, a->data(), &lda_i, ws->S.data(), ws->U.data(), &ldu_i, ws->Vh.data(),
                     &ldvt_i, ws->work.data(), &lwork_i, ws->rwork.data(), ws->iwork.data(), &info);
    if (info != 0) {
        throw lapack::detail::SvdError("gesdd", jobz, m, n, info);
    }
}
}  // namespace mindquantum::sim::mps::detail::linalg

#endif  // INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_LAPACK_SVD_HPP
