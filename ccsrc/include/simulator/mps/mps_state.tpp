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

#ifndef INCLUDE_SIMULATOR_MPS_MPS_STATE_TPP
#define INCLUDE_SIMULATOR_MPS_MPS_STATE_TPP

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "core/sparse/algo.h"
#include "core/utils.h"
#include "simulator/mps/compiler/mps_compiler.h"

namespace mindquantum::sim::mps::detail {
namespace detail {
template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using RowVec = Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>;

template <typename calc_t>
inline calc_t SafeInv(calc_t x, calc_t floor) {
    if (x < floor) {
        return static_cast<calc_t>(1) / floor;
    }
    return static_cast<calc_t>(1) / x;
}
}  // namespace detail

template <typename qs_policy_t_>
MPSState<qs_policy_t_>::MPSState(qbit_t n_qubits, unsigned seed, uint32_t max_bond_dimension)
    : n_qubits_(n_qubits), seed_(seed), max_bond_dimension_(max_bond_dimension), rnd_eng_(seed) {
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
    Reset();
}

template <typename qs_policy_t_>
MPSState<qs_policy_t_>::MPSState(const MPSState& other)
    : n_qubits_(other.n_qubits_)
    , seed_(other.seed_)
    , max_bond_dimension_(other.max_bond_dimension_)
    , rnd_eng_(other.seed_)
    , lambdas_(other.lambdas_)
    , gammas_(other.gammas_)
    , pos_of_logical_(other.pos_of_logical_)
    , logical_at_pos_(other.logical_at_pos_) {
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::operator=(const MPSState& other) -> derived_t& {
    if (this == &other) {
        return *this;
    }
    n_qubits_ = other.n_qubits_;
    seed_ = other.seed_;
    max_bond_dimension_ = other.max_bond_dimension_;
    rnd_eng_ = RndEngine(seed_);
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));

    lambdas_ = other.lambdas_;
    gammas_ = other.gammas_;
    pos_of_logical_ = other.pos_of_logical_;
    logical_at_pos_ = other.logical_at_pos_;
    return *this;
}

template <typename qs_policy_t_>
MPSState<qs_policy_t_>::MPSState(MPSState&& other) noexcept
    : n_qubits_(other.n_qubits_)
    , seed_(other.seed_)
    , max_bond_dimension_(other.max_bond_dimension_)
    , rnd_eng_(other.seed_)
    , lambdas_(std::move(other.lambdas_))
    , gammas_(std::move(other.gammas_))
    , pos_of_logical_(std::move(other.pos_of_logical_))
    , logical_at_pos_(std::move(other.logical_at_pos_)) {
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::operator=(MPSState&& other) noexcept -> derived_t& {
    if (this == &other) {
        return *this;
    }
    n_qubits_ = other.n_qubits_;
    seed_ = other.seed_;
    max_bond_dimension_ = other.max_bond_dimension_;
    rnd_eng_ = RndEngine(seed_);
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
    lambdas_ = std::move(other.lambdas_);
    gammas_ = std::move(other.gammas_);
    pos_of_logical_ = std::move(other.pos_of_logical_);
    logical_at_pos_ = std::move(other.logical_at_pos_);
    return *this;
}

template <typename qs_policy_t_>
tensor::TDtype MPSState<qs_policy_t_>::DType() {
    return qs_policy_t::kDType;
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::SetSeed(unsigned new_seed) {
    seed_ = new_seed;
    rnd_eng_ = RndEngine(new_seed);
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::InitZeroState_() {
    lambdas_.assign(static_cast<size_t>(n_qubits_) + 1, VT<calc_type>{static_cast<calc_type>(1)});
    gammas_.assign(static_cast<size_t>(n_qubits_), GammaSite{});
    for (qbit_t i = 0; i < n_qubits_; ++i) {
        auto& g = gammas_.at(static_cast<size_t>(i));
        g.dl = 1;
        g.dr = 1;
        g.g0.assign(1, complex_t{1, 0});
        g.g1.assign(1, complex_t{0, 0});
    }
    pos_of_logical_.resize(static_cast<size_t>(n_qubits_));
    logical_at_pos_.resize(static_cast<size_t>(n_qubits_));
    for (qbit_t q = 0; q < n_qubits_; ++q) {
        pos_of_logical_.at(static_cast<size_t>(q)) = q;
        logical_at_pos_.at(static_cast<size_t>(q)) = q;
    }
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::Reset() {
    InitZeroState_();
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::Display(qbit_t qubits_limit) const {
    (void)qubits_limit;
    // Keep output minimal (similar to other backends).
    std::cout << "mqmps state: n_qubits=" << n_qubits_ << ", max_bond_dimension=" << max_bond_dimension_ << std::endl;
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::ContractToStateVectorPhysicalOrder_() const -> VT<complex_t> {
    using Mat = detail::Mat<complex_t>;
    const size_t n = static_cast<size_t>(n_qubits_);
    if (n == 0) {
        return VT<complex_t>{complex_t{1, 0}};
    }

    Mat acc(1, 1);
    acc(0, 0) = complex_t{1, 0};

    for (qbit_t p = 0; p < n_qubits_; ++p) {
        const auto& lam_next = lambdas_.at(static_cast<size_t>(p + 1));
        const size_t dl = lambdas_.at(static_cast<size_t>(p)).size();
        const size_t dr = lam_next.size();
        const auto& site = gammas_.at(static_cast<size_t>(p));

        if (site.dl != dl || site.dr != dr) {
            throw std::runtime_error("mqmps: inconsistent gamma/lambda dimensions.");
        }

        Eigen::Map<const Mat> g0(site.g0.data(), static_cast<Eigen::Index>(site.dl), static_cast<Eigen::Index>(site.dr));
        Eigen::Map<const Mat> g1(site.g1.data(), static_cast<Eigen::Index>(site.dl), static_cast<Eigen::Index>(site.dr));

        Mat a0 = acc * g0;
        Mat a1 = acc * g1;

        // Scale columns by lambda[p+1]
        for (size_t j = 0; j < dr; ++j) {
            a0.col(static_cast<Eigen::Index>(j)) *= lam_next[j];
            a1.col(static_cast<Eigen::Index>(j)) *= lam_next[j];
        }

        Mat next(2 * acc.rows(), a0.cols());
        next.topRows(acc.rows()) = a0;
        next.bottomRows(acc.rows()) = a1;
        acc = std::move(next);
    }

    if (acc.cols() != 1) {
        throw std::runtime_error("mqmps: final bond dimension is not 1; cannot convert to full state vector.");
    }
    VT<complex_t> out(static_cast<size_t>(acc.rows()));
    for (Eigen::Index i = 0; i < acc.rows(); ++i) {
        out.at(static_cast<size_t>(i)) = acc(i, 0);
    }
    return out;
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::PermutePhysicalToLogical_(const VT<complex_t>& phys) const -> VT<complex_t> {
    const size_t n = static_cast<size_t>(n_qubits_);
    const size_t dim = static_cast<size_t>(1ULL) << n;
    if (phys.size() != dim) {
        throw std::runtime_error("mqmps: invalid physical state vector size.");
    }
    VT<complex_t> logical(dim);
    for (size_t i_log = 0; i_log < dim; ++i_log) {
        size_t i_phys = 0;
        for (qbit_t q = 0; q < n_qubits_; ++q) {
            const auto bit = (i_log >> static_cast<size_t>(q)) & 1U;
            const auto p = static_cast<size_t>(pos_of_logical_.at(static_cast<size_t>(q)));
            i_phys |= (bit << p);
        }
        logical.at(i_log) = phys.at(i_phys);
    }
    return logical;
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::GetQS() const -> VT<py_qs_data_t> {
    auto phys = ContractToStateVectorPhysicalOrder_();
    auto logical = PermutePhysicalToLogical_(phys);
    return logical;
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::SetFromStateVectorPhysical_(const VT<complex_t>& vec_phys) {
    using Mat = detail::Mat<complex_t>;
    using VecR = detail::Vec<calc_type>;
    constexpr calc_type eps_rel = qs_policy_t::kEpsRel;
    constexpr calc_type lambda_floor = qs_policy_t::kLambdaFloor;

    if (n_qubits_ == 0) {
        InitZeroState_();
        return;
    }
    const size_t dim = static_cast<size_t>(1ULL) << static_cast<size_t>(n_qubits_);
    if (vec_phys.size() != dim) {
        throw std::runtime_error("mqmps: SetQS vector size mismatch.");
    }

    // Initialize containers.
    lambdas_.assign(static_cast<size_t>(n_qubits_) + 1, VT<calc_type>{});
    gammas_.assign(static_cast<size_t>(n_qubits_), GammaSite{});
    lambdas_.at(0) = VT<calc_type>{static_cast<calc_type>(1)};

    // B has shape (D_i, 2^(n-i)), with i=0 -> (1, 2^n)
    Mat B(1, static_cast<Eigen::Index>(dim));
    for (size_t i = 0; i < dim; ++i) {
        B(0, static_cast<Eigen::Index>(i)) = vec_phys.at(i);
    }

    for (qbit_t i = 0; i < n_qubits_; ++i) {
        const size_t D_i = lambdas_.at(static_cast<size_t>(i)).size();
        const size_t remaining = static_cast<size_t>(n_qubits_ - i);
        const size_t cols = static_cast<size_t>(1ULL) << remaining;
        const size_t cols_rest = cols / 2;

        if (static_cast<size_t>(B.rows()) != D_i || static_cast<size_t>(B.cols()) != cols) {
            throw std::runtime_error("mqmps: internal SetQS reshape mismatch.");
        }

        Mat M(static_cast<Eigen::Index>(2 * D_i), static_cast<Eigen::Index>(cols_rest));
        for (size_t cr = 0; cr < cols_rest; ++cr) {
            const size_t c0 = 2 * cr;
            const size_t c1 = 2 * cr + 1;
            M.block(0, static_cast<Eigen::Index>(cr), static_cast<Eigen::Index>(D_i), 1) =
                B.col(static_cast<Eigen::Index>(c0));
            M.block(static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(cr), static_cast<Eigen::Index>(D_i), 1) =
                B.col(static_cast<Eigen::Index>(c1));
        }

        Eigen::BDCSVD<Mat> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto s_all = svd.singularValues();  // real
        const auto r_full = static_cast<size_t>(s_all.size());
        size_t r_keep = r_full;
        if (max_bond_dimension_ != 0) {
            r_keep = std::min(r_keep, static_cast<size_t>(max_bond_dimension_));
        }
        if (r_keep == 0) {
            r_keep = 1;
        }
        if (r_keep > 1 && s_all(0) > 0) {
            for (size_t k = 0; k < r_keep; ++k) {
                if (s_all(static_cast<Eigen::Index>(k)) / s_all(0) < eps_rel) {
                    r_keep = std::max<size_t>(1, k);
                    break;
                }
            }
        }

        // Copy + normalize kept singular values.
        VT<calc_type> s_kept(r_keep);
        calc_type s_norm2 = 0;
        for (size_t k = 0; k < r_keep; ++k) {
            s_kept[k] = s_all(static_cast<Eigen::Index>(k));
            s_norm2 += s_kept[k] * s_kept[k];
        }
        const calc_type s_norm = std::sqrt(s_norm2);
        if (s_norm > 0) {
            for (auto& v : s_kept) {
                v /= s_norm;
            }
        }
        lambdas_.at(static_cast<size_t>(i + 1)) = std::move(s_kept);

        const size_t D_next = lambdas_.at(static_cast<size_t>(i + 1)).size();
        auto& site = gammas_.at(static_cast<size_t>(i));
        site.dl = D_i;
        site.dr = D_next;
        site.g0.resize(D_i * D_next);
        site.g1.resize(D_i * D_next);

        const auto U = svd.matrixU().leftCols(static_cast<Eigen::Index>(D_next));
        for (size_t a = 0; a < D_i; ++a) {
            const calc_type inv = detail::SafeInv(lambdas_.at(static_cast<size_t>(i)).at(a), lambda_floor);
            for (size_t k = 0; k < D_next; ++k) {
                site.g0.at(a * D_next + k) = U(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(k)) * inv;
                site.g1.at(a * D_next + k) =
                    U(static_cast<Eigen::Index>(a + D_i), static_cast<Eigen::Index>(k)) * inv;
            }
        }

        // Build next B = diag(S) * Vh
        const auto V = svd.matrixV().leftCols(static_cast<Eigen::Index>(D_next));  // (cols_rest x D_next)
        Mat B_next(static_cast<Eigen::Index>(D_next), static_cast<Eigen::Index>(cols_rest));
        B_next.setZero();
        for (size_t k = 0; k < D_next; ++k) {
            const calc_type s_k = lambdas_.at(static_cast<size_t>(i + 1)).at(k);
            B_next.row(static_cast<Eigen::Index>(k)) = s_k * V.col(static_cast<Eigen::Index>(k)).adjoint();
        }
        B = std::move(B_next);
    }

    // Ensure mapping is identity after SetQS.
    pos_of_logical_.resize(static_cast<size_t>(n_qubits_));
    logical_at_pos_.resize(static_cast<size_t>(n_qubits_));
    for (qbit_t q = 0; q < n_qubits_; ++q) {
        pos_of_logical_.at(static_cast<size_t>(q)) = q;
        logical_at_pos_.at(static_cast<size_t>(q)) = q;
    }
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::SetFromStateVectorLogical_(const VT<complex_t>& vec) {
    // Reset internal mapping and treat vec as physical ordering.
    SetFromStateVectorPhysical_(vec);
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::SetQS(const VT<py_qs_data_t>& qs) {
    VT<complex_t> vec(qs.begin(), qs.end());
    // Normalize defensively.
    calc_type norm2 = 0;
    for (const auto& v : vec) {
        norm2 += std::norm(v);
    }
    const calc_type norm = std::sqrt(norm2);
    if (norm == 0) {
        throw std::runtime_error("mqmps: cannot set zero state vector.");
    }
    for (auto& v : vec) {
        v /= norm;
    }
    SetFromStateVectorLogical_(vec);
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::Apply1QOnPhysical_(qbit_t pos, const std::array<complex_t, 4>& u) {
    auto& site = gammas_.at(static_cast<size_t>(pos));
    const auto u00 = u[0];
    const auto u01 = u[1];
    const auto u10 = u[2];
    const auto u11 = u[3];
    const size_t size = site.g0.size();
    for (size_t i = 0; i < size; ++i) {
        const auto a = site.g0[i];
        const auto b = site.g1[i];
        site.g0[i] = u00 * a + u01 * b;
        site.g1[i] = u10 * a + u11 * b;
    }
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::Apply2QOnBond_(qbit_t bond, const std::array<complex_t, 16>& u) {
    using Mat = detail::Mat<complex_t>;
    constexpr calc_type eps_rel = qs_policy_t::kEpsRel;
    constexpr calc_type lambda_floor = qs_policy_t::kLambdaFloor;

    const qbit_t i = bond;
    const qbit_t j = bond + 1;
    const size_t D_L = lambdas_.at(static_cast<size_t>(i)).size();
    const size_t D_M = lambdas_.at(static_cast<size_t>(i + 1)).size();
    const size_t D_R = lambdas_.at(static_cast<size_t>(i + 2)).size();

    auto& site_i = gammas_.at(static_cast<size_t>(i));
    auto& site_j = gammas_.at(static_cast<size_t>(j));
    if (site_i.dl != D_L || site_i.dr != D_M || site_j.dl != D_M || site_j.dr != D_R) {
        throw std::runtime_error("mqmps: inconsistent bond dimensions for 2-site update.");
    }

    thread_local Mat L;
    thread_local Mat R;
    thread_local Mat Theta;

    L.resize(static_cast<Eigen::Index>(2 * D_L), static_cast<Eigen::Index>(D_M));
    R.resize(static_cast<Eigen::Index>(D_M), static_cast<Eigen::Index>(2 * D_R));

    // Build L
    for (size_t a = 0; a < D_L; ++a) {
        const calc_type lam_left = lambdas_.at(static_cast<size_t>(i)).at(a);
        for (size_t m = 0; m < D_M; ++m) {
            const calc_type lam_mid = lambdas_.at(static_cast<size_t>(i + 1)).at(m);
            const auto scale = lam_left * lam_mid;
            L(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(m)) = site_i.g0.at(a * D_M + m) * scale;
            L(static_cast<Eigen::Index>(a + D_L), static_cast<Eigen::Index>(m)) = site_i.g1.at(a * D_M + m) * scale;
        }
    }

    // Build R
    for (size_t m = 0; m < D_M; ++m) {
        for (size_t r = 0; r < D_R; ++r) {
            const calc_type lam_right = lambdas_.at(static_cast<size_t>(i + 2)).at(r);
            R(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(r)) = site_j.g0.at(m * D_R + r) * lam_right;
            R(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(r + D_R)) = site_j.g1.at(m * D_R + r) * lam_right;
        }
    }

    Theta.resize(static_cast<Eigen::Index>(2 * D_L), static_cast<Eigen::Index>(2 * D_R));
    Theta.noalias() = L * R;

    // Apply 4x4 gate to physical indices (unrolled).
    for (size_t a = 0; a < D_L; ++a) {
        for (size_t r = 0; r < D_R; ++r) {
            const auto t00 = Theta(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(r));
            const auto t01 = Theta(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(r + D_R));
            const auto t10 = Theta(static_cast<Eigen::Index>(a + D_L), static_cast<Eigen::Index>(r));
            const auto t11 = Theta(static_cast<Eigen::Index>(a + D_L), static_cast<Eigen::Index>(r + D_R));

            const auto v00 = u[0] * t00 + u[1] * t01 + u[2] * t10 + u[3] * t11;
            const auto v01 = u[4] * t00 + u[5] * t01 + u[6] * t10 + u[7] * t11;
            const auto v10 = u[8] * t00 + u[9] * t01 + u[10] * t10 + u[11] * t11;
            const auto v11 = u[12] * t00 + u[13] * t01 + u[14] * t10 + u[15] * t11;

            Theta(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(r)) = v00;
            Theta(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(r + D_R)) = v01;
            Theta(static_cast<Eigen::Index>(a + D_L), static_cast<Eigen::Index>(r)) = v10;
            Theta(static_cast<Eigen::Index>(a + D_L), static_cast<Eigen::Index>(r + D_R)) = v11;
        }
    }

    Eigen::BDCSVD<Mat> svd(Theta, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto s_all = svd.singularValues();
    const size_t r_full = static_cast<size_t>(s_all.size());
    size_t r_keep = r_full;
    if (max_bond_dimension_ != 0) {
        r_keep = std::min(r_keep, static_cast<size_t>(max_bond_dimension_));
    }
    if (r_keep == 0) {
        r_keep = 1;
    }
    if (r_keep > 1 && s_all(0) > 0) {
        for (size_t k = 0; k < r_keep; ++k) {
            if (s_all(static_cast<Eigen::Index>(k)) / s_all(0) < eps_rel) {
                r_keep = std::max<size_t>(1, k);
                break;
            }
        }
    }

    VT<calc_type> s_kept(r_keep);
    calc_type norm2 = 0;
    for (size_t k = 0; k < r_keep; ++k) {
        s_kept[k] = s_all(static_cast<Eigen::Index>(k));
        norm2 += s_kept[k] * s_kept[k];
    }
    const calc_type norm = std::sqrt(norm2);
    if (norm > 0) {
        for (auto& v : s_kept) {
            v /= norm;
        }
    }

    lambdas_.at(static_cast<size_t>(i + 1)) = std::move(s_kept);

    const size_t D_new = lambdas_.at(static_cast<size_t>(i + 1)).size();
    const auto U = svd.matrixU().leftCols(static_cast<Eigen::Index>(D_new));
    const auto V = svd.matrixV().leftCols(static_cast<Eigen::Index>(D_new));  // (2*D_R x D_new)

    // Update left site gamma (D_L x D_new)
    site_i.dr = D_new;
    site_i.g0.resize(D_L * D_new);
    site_i.g1.resize(D_L * D_new);
    for (size_t a = 0; a < D_L; ++a) {
        const calc_type inv = detail::SafeInv(lambdas_.at(static_cast<size_t>(i)).at(a), lambda_floor);
        for (size_t k = 0; k < D_new; ++k) {
            site_i.g0.at(a * D_new + k) = U(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(k)) * inv;
            site_i.g1.at(a * D_new + k) = U(static_cast<Eigen::Index>(a + D_L), static_cast<Eigen::Index>(k)) * inv;
        }
    }

    // Update right site gamma (D_new x D_R) from Vh blocks, divide by lambda[i+2]
    site_j.dl = D_new;
    site_j.g0.resize(D_new * D_R);
    site_j.g1.resize(D_new * D_R);
    for (size_t k = 0; k < D_new; ++k) {
        for (size_t r = 0; r < D_R; ++r) {
            const calc_type inv = detail::SafeInv(lambdas_.at(static_cast<size_t>(i + 2)).at(r), lambda_floor);
            // Vh(k, r) = conj(V(r, k))
            site_j.g0.at(k * D_R + r) = std::conj(V(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(k))) * inv;
            site_j.g1.at(k * D_R + r) =
                std::conj(V(static_cast<Eigen::Index>(r + D_R), static_cast<Eigen::Index>(k))) * inv;
        }
    }
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::ApplyCircuitCompiled_(const circuit_t& circ, const parameter::ParameterResolver& pr,
                                                   bool diff) -> std::map<std::string, int> {
    compiler::MPSCompiler<calc_type> comp(n_qubits_);
    auto program = comp.Compile(circ, pr, pos_of_logical_, logical_at_pos_, diff);

    std::map<std::string, int> meas_result;

    // Execute unitary stages first.
    size_t first_meas_stage = program.stages.size();
    for (size_t s = 0; s < program.stages.size(); ++s) {
        if (!program.stages[s].meas.empty()) {
            first_meas_stage = s;
            break;
        }
    }
    for (size_t s = 0; s < first_meas_stage; ++s) {
        const auto& stage = program.stages[s];
        for (const auto& op : stage.op1) {
            Apply1QOnPhysical_(op.pos, op.mat);
        }
        // 2Q ops are disjoint in a stage; parallelize when it helps.
        const auto& ops2 = stage.op2;
        THRESHOLD_OMP_FOR(
            ops2.size(), 2, for (omp::idx_t idx = 0; idx < static_cast<omp::idx_t>(ops2.size()); ++idx) {
                const auto& op = ops2.at(static_cast<size_t>(idx));
                Apply2QOnBond_(op.bond, op.mat);
            })
    }

    // Update mapping to final mapping before measurement (needed to interpret physical positions).
    pos_of_logical_ = std::move(program.pos_of_logical_final);
    logical_at_pos_ = std::move(program.logical_at_pos_final);

    // Measurement is currently only supported as a final stage sequence.
    if (first_meas_stage < program.stages.size()) {
        for (size_t s = first_meas_stage; s < program.stages.size(); ++s) {
            if (s != first_meas_stage) {
                throw std::runtime_error("mqmps: mid-circuit measurement is not supported.");
            }
            const auto& stage = program.stages[s];
            if (stage.meas.empty()) {
                continue;
            }

            // One-shot measurement of the final state: sample a full basis index and project.
            auto psi = GetQS();  // logical ordering
            const size_t dim = psi.size();
            if (dim == 0) {
                throw std::runtime_error("mqmps: invalid state vector size.");
            }
            VT<double> cum_prob(dim);
            double acc = 0.0;
            for (size_t i = 0; i < dim; ++i) {
                acc += static_cast<double>(std::norm(psi[i]));
                cum_prob[i] = acc;
            }
            if (acc == 0.0) {
                throw std::runtime_error("mqmps: zero-norm state in measurement.");
            }
            for (auto& v : cum_prob) {
                v /= acc;
            }
            const double r = rng_();
            auto it = std::lower_bound(cum_prob.begin(), cum_prob.end(), r);
            const size_t sampled = static_cast<size_t>(std::distance(cum_prob.begin(), it));

            // Record measured bits (in logical qubit indices).
            struct ResetInfo {
                qbit_t q = 0;
                bool do_reset = false;
                index_t reset_to = 0;
                int outcome = 0;
            };
            VT<ResetInfo> resets;
            resets.reserve(stage.meas.size());
            for (const auto& m : stage.meas) {
                const auto q = logical_at_pos_.at(static_cast<size_t>(m.pos));
                const int bit = static_cast<int>((sampled >> static_cast<size_t>(q)) & 1U);
                meas_result[m.key] = bit;
                resets.push_back(ResetInfo{q, m.reset, m.reset_to, bit});
            }

            // Collapse: keep only amplitudes matching all measured outcomes.
            for (size_t i = 0; i < dim; ++i) {
                bool keep = true;
                for (const auto& info : resets) {
                    const int b = static_cast<int>((i >> static_cast<size_t>(info.q)) & 1U);
                    if (b != info.outcome) {
                        keep = false;
                        break;
                    }
                }
                if (!keep) {
                    psi[i] = complex_t{0, 0};
                }
            }
            // Renormalize.
            double new_norm2 = 0.0;
            for (const auto& v : psi) {
                new_norm2 += static_cast<double>(std::norm(v));
            }
            if (new_norm2 == 0.0) {
                throw std::runtime_error("mqmps: collapse produced zero state.");
            }
            const double inv_norm = 1.0 / std::sqrt(new_norm2);
            for (auto& v : psi) {
                v *= static_cast<calc_type>(inv_norm);
            }

            // Apply post-measure reset.
            for (const auto& info : resets) {
                if (!info.do_reset) {
                    continue;
                }
                if (static_cast<index_t>(info.outcome) == info.reset_to) {
                    continue;
                }
                const size_t mask = static_cast<size_t>(1ULL) << static_cast<size_t>(info.q);
                for (size_t i = 0; i < dim; ++i) {
                    if ((i & mask) == 0) {
                        std::swap(psi[i], psi[i | mask]);
                    }
                }
            }

            // Rebuild MPS from the collapsed logical state (resets mapping to identity).
            SetFromStateVectorLogical_(psi);
        }
    }

    return meas_result;
}

template <typename qs_policy_t_>
index_t MPSState<qs_policy_t_>::ApplyGate(const std::shared_ptr<BasicGate>& gate, const parameter::ParameterResolver& pr,
                                          bool diff) {
    if (gate->GetID() == GateID::M) {
        auto res = ApplyCircuit({gate}, pr);
        if (res.empty()) {
            return 2;
        }
        return static_cast<index_t>(res.begin()->second);
    }
    ApplyCircuit({gate}, pr);
    return 2;
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::ApplyCircuit(const circuit_t& circ, const parameter::ParameterResolver& pr)
    -> std::map<std::string, int> {
    return ApplyCircuitCompiled_(circ, pr, false);
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::Sampling(const circuit_t& circ, const parameter::ParameterResolver& pr, size_t shots,
                                      const MST<size_t>& key_map, unsigned seed) const -> VT<unsigned> {
    const auto key_size = key_map.size();
    VT<unsigned> res(shots * key_size);
    RndEngine rnd_eng(seed);
    std::uniform_real_distribution<double> dist(0., 1.);
    std::function<double()> rng = std::bind(dist, std::ref(rnd_eng));
    for (size_t i = 0; i < shots; ++i) {
        auto sim = derived_t(*this);
        sim.SetSeed(static_cast<unsigned>(rng() * (1 << 20)));
        auto r0 = sim.ApplyCircuit(circ, pr);
        for (const auto& [k, v] : r0) {
            res[i * key_size + key_map.at(k)] = static_cast<unsigned>(v);
        }
    }
    return res;
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::SamplingMeasurementEndingWithoutNoise(const circuit_t& circ, const parameter::ParameterResolver& pr,
                                                                   size_t shots, const MST<size_t>& key_map,
                                                                   unsigned seed) const -> VT<unsigned> {
    using Mat = detail::Mat<complex_t>;
    using RowVec = detail::RowVec<complex_t>;

    auto sim = derived_t(*this);
    sim.SetSeed(seed);

    // Split measurement and unitary parts.
    VT<int> already_measured(static_cast<size_t>(n_qubits_), 0);
    VT<qbit_t> m_qids;
    std::map<int, std::string> qid_maps;
    circuit_t other_circ;
    bool seen_measure = false;
    for (const auto& g : circ) {
        if (g->GetID() == GateID::M) {
            seen_measure = true;
            auto m_qid = g->GetObjQubits().at(0);
            if (already_measured.at(static_cast<size_t>(m_qid)) != 0) {
                throw std::runtime_error("Quantum circuit is not a measurement ending circuit.");
            }
            already_measured.at(static_cast<size_t>(m_qid)) = 1;
            m_qids.push_back(m_qid);
            qid_maps[m_qid] = static_cast<MeasureGate*>(g.get())->Name();
        } else {
            if (seen_measure) {
                throw std::runtime_error("mqmps: mid-circuit measurement is not supported for fast sampling.");
            }
            other_circ.push_back(g);
        }
    }
    sim.ApplyCircuit(other_circ, pr);

    // Precompute right environments R[i] (D_i x D_i), where D_i = |Lambda[i]|.
    VT<Mat> R_env(static_cast<size_t>(sim.n_qubits_) + 1);
    R_env.at(static_cast<size_t>(sim.n_qubits_)) = Mat(1, 1);
    R_env.at(static_cast<size_t>(sim.n_qubits_))(0, 0) = complex_t{1, 0};
    for (qbit_t i = sim.n_qubits_ - 1; i >= 0; --i) {
        const auto D_i = sim.lambdas_.at(static_cast<size_t>(i)).size();
        const auto D_next = sim.lambdas_.at(static_cast<size_t>(i + 1)).size();
        const auto& lam_next = sim.lambdas_.at(static_cast<size_t>(i + 1));
        const auto& site = sim.gammas_.at(static_cast<size_t>(i));
        if (site.dl != D_i || site.dr != D_next) {
            throw std::runtime_error("mqmps: inconsistent dimensions in sampling env.");
        }
        Eigen::Map<const Mat> g0(site.g0.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        Eigen::Map<const Mat> g1(site.g1.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        const auto& Rn = R_env.at(static_cast<size_t>(i + 1));
        Mat Rscaled = Rn;
        // Rscaled = diag(lambda_next) * Rn * diag(lambda_next)
        for (size_t k = 0; k < D_next; ++k) {
            Rscaled.row(static_cast<Eigen::Index>(k)) *= lam_next[k];
            Rscaled.col(static_cast<Eigen::Index>(k)) *= lam_next[k];
        }
        Mat Ri = g0 * Rscaled * g0.adjoint() + g1 * Rscaled * g1.adjoint();
        R_env.at(static_cast<size_t>(i)) = std::move(Ri);
    }

    const auto key_size = key_map.size();
    VT<unsigned> out(shots * key_size);

    for (size_t shot = 0; shot < shots; ++shot) {
        // Sample full register bits in physical site order.
        VT<int> bits(static_cast<size_t>(sim.n_qubits_), 0);
        RowVec x(1);
        x(0) = complex_t{1, 0};
        for (qbit_t i = 0; i < sim.n_qubits_; ++i) {
            const auto D_i = sim.lambdas_.at(static_cast<size_t>(i)).size();
            const auto D_next = sim.lambdas_.at(static_cast<size_t>(i + 1)).size();
            const auto& lam_next = sim.lambdas_.at(static_cast<size_t>(i + 1));
            const auto& site = sim.gammas_.at(static_cast<size_t>(i));
            if (site.dl != D_i || site.dr != D_next) {
                throw std::runtime_error("mqmps: inconsistent dimensions in sampling.");
            }
            Eigen::Map<const Mat> g0(site.g0.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
            Eigen::Map<const Mat> g1(site.g1.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
            if (x.cols() != static_cast<Eigen::Index>(D_i)) {
                throw std::runtime_error("mqmps: internal sampling left-vector dim mismatch.");
            }
            RowVec y0 = x * g0;
            RowVec y1 = x * g1;
            for (size_t k = 0; k < D_next; ++k) {
                y0(static_cast<Eigen::Index>(k)) *= lam_next[k];
                y1(static_cast<Eigen::Index>(k)) *= lam_next[k];
            }
            const auto& Rn = R_env.at(static_cast<size_t>(i + 1));
            const complex_t p0 = (y0 * (Rn * y0.adjoint()))(0, 0);
            const complex_t p1 = (y1 * (Rn * y1.adjoint()))(0, 0);
            const double p0r = std::max(0.0, static_cast<double>(std::real(p0)));
            const double p1r = std::max(0.0, static_cast<double>(std::real(p1)));
            const double denom = p0r + p1r;
            if (denom <= 0) {
                throw std::runtime_error("mqmps: invalid probability in sampling.");
            }
            const double prob0 = p0r / denom;
            const bool bit0 = (sim.rng_() < prob0);
            bits.at(static_cast<size_t>(i)) = bit0 ? 0 : 1;
            x = bit0 ? y0 : y1;
        }

        // Report only measured logical qubits.
        for (auto q : m_qids) {
            const auto pos = sim.pos_of_logical_.at(static_cast<size_t>(q));
            const auto bit = bits.at(static_cast<size_t>(pos));
            out[shot * key_size + key_map.at(qid_maps.at(q))] = static_cast<unsigned>(bit);
        }
    }

    return out;
}

template <typename qs_policy_t_>
VVT<typename MPSState<qs_policy_t_>::py_qs_data_t> MPSState<qs_policy_t_>::GetReducedDensityMatrix(
    const qbits_t& kept_qubits) const {
    using Mat = detail::Mat<complex_t>;
    const auto psi = GetQS();  // logical order
    const size_t n = static_cast<size_t>(n_qubits_);
    const size_t k = kept_qubits.size();
    if (k == 0 || k >= n) {
        throw std::runtime_error("mqmps: kept_qubits must be between 1 and n_qubits-1.");
    }
    const size_t dim_full = static_cast<size_t>(1ULL) << n;
    const size_t dim_keep = static_cast<size_t>(1ULL) << k;
    const size_t dim_env = static_cast<size_t>(1ULL) << (n - k);

    // Build the list of environment qubits.
    VT<qbit_t> kept = kept_qubits;
    std::sort(kept.begin(), kept.end());
    kept.erase(std::unique(kept.begin(), kept.end()), kept.end());
    if (kept.size() != k) {
        throw std::runtime_error("mqmps: kept_qubits contains duplicates.");
    }
    VT<qbit_t> env;
    env.reserve(n - k);
    for (qbit_t q = 0; q < n_qubits_; ++q) {
        if (!std::binary_search(kept.begin(), kept.end(), q)) {
            env.push_back(q);
        }
    }

    Mat psi_mat(static_cast<Eigen::Index>(dim_keep), static_cast<Eigen::Index>(dim_env));
    psi_mat.setZero();
    for (size_t idx = 0; idx < dim_full; ++idx) {
        size_t a = 0;
        size_t e = 0;
        for (size_t j = 0; j < kept.size(); ++j) {
            const auto q = static_cast<size_t>(kept[j]);
            a |= (((idx >> q) & 1U) << j);
        }
        for (size_t j = 0; j < env.size(); ++j) {
            const auto q = static_cast<size_t>(env[j]);
            e |= (((idx >> q) & 1U) << j);
        }
        psi_mat(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(e)) = psi.at(idx);
    }

    Mat rho = psi_mat * psi_mat.adjoint();
    VVT<complex_t> out(static_cast<size_t>(rho.rows()), VT<complex_t>(static_cast<size_t>(rho.cols())));
    for (Eigen::Index r = 0; r < rho.rows(); ++r) {
        for (Eigen::Index c = 0; c < rho.cols(); ++c) {
            out.at(static_cast<size_t>(r)).at(static_cast<size_t>(c)) = rho(r, c);
        }
    }
    return out;
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::ExpectationPauliTerms_(const VT<PauliTerm<calc_type>>& terms) const -> py_qs_data_t {
    using Mat = detail::Mat<complex_t>;
    constexpr complex_t I{0, 1};
    py_qs_data_t out{0, 0};

    for (const auto& [pwords, coeff] : terms) {
        // Local operators per physical site (default identity).
        VT<std::array<complex_t, 4>> ops(static_cast<size_t>(n_qubits_), {complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{1, 0}});
        for (const auto& [q, p] : pwords) {
            const auto pos = pos_of_logical_.at(static_cast<size_t>(q));
            switch (p) {
                case 'X':
                    ops.at(static_cast<size_t>(pos)) = {complex_t{0, 0}, complex_t{1, 0}, complex_t{1, 0}, complex_t{0, 0}};
                    break;
                case 'Y':
                    ops.at(static_cast<size_t>(pos)) = {complex_t{0, 0}, -I, I, complex_t{0, 0}};
                    break;
                case 'Z':
                    ops.at(static_cast<size_t>(pos)) = {complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{-1, 0}};
                    break;
                default:
                    throw std::runtime_error("mqmps: unknown Pauli word.");
            }
        }

        Mat E(1, 1);
        E(0, 0) = complex_t{1, 0};
        for (qbit_t i = 0; i < n_qubits_; ++i) {
            const auto D_i = lambdas_.at(static_cast<size_t>(i)).size();
            const auto D_next = lambdas_.at(static_cast<size_t>(i + 1)).size();
            const auto& lam_next = lambdas_.at(static_cast<size_t>(i + 1));
            const auto& site = gammas_.at(static_cast<size_t>(i));
            Eigen::Map<const Mat> g0(site.g0.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
            Eigen::Map<const Mat> g1(site.g1.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));

            const auto o = ops.at(static_cast<size_t>(i));
            const complex_t o00 = o[0];
            const complex_t o01 = o[1];
            const complex_t o10 = o[2];
            const complex_t o11 = o[3];

            Mat t00 = g0.adjoint() * E * g0;
            Mat t01 = g0.adjoint() * E * g1;
            Mat t10 = g1.adjoint() * E * g0;
            Mat t11 = g1.adjoint() * E * g1;
            Mat X = t00 * o00 + t01 * o01 + t10 * o10 + t11 * o11;

            // E_next = diag(lambda_next) * X * diag(lambda_next)
            Mat E_next = X;
            for (size_t k = 0; k < D_next; ++k) {
                E_next.row(static_cast<Eigen::Index>(k)) *= lam_next[k];
                E_next.col(static_cast<Eigen::Index>(k)) *= lam_next[k];
            }
            E = std::move(E_next);
        }
        out += static_cast<calc_type>(coeff) * E(0, 0);
    }
    return out;
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::ExpectationSparse_(const Hamiltonian<calc_type>& ham) const -> py_qs_data_t {
    auto psi = GetQS();  // logical order
    if (ham.how_to_ == BACKEND) {
        return sparse::ExpectationOfCsr<calc_type, calc_type>(ham.ham_sparse_main_, ham.ham_sparse_second_,
                                                              reinterpret_cast<calc_type*>(psi.data()),
                                                              reinterpret_cast<calc_type*>(psi.data()));
    }
    if (ham.how_to_ == FRONTEND) {
        return sparse::ExpectationOfCsr<calc_type, calc_type>(ham.ham_sparse_main_,
                                                              reinterpret_cast<calc_type*>(psi.data()),
                                                              reinterpret_cast<calc_type*>(psi.data()));
    }
    throw std::runtime_error("mqmps: sparse expectation requires a sparse Hamiltonian.");
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                            const parameter::ParameterResolver& pr) const -> py_qs_data_t {
    auto sub_seed = static_cast<unsigned>(rng_() * (1 << 20));
    auto ket = derived_t(*this);
    ket.SetSeed(sub_seed);
    ket.ApplyCircuit(circ, pr);
    if (ham.how_to_ == ORIGIN) {
        return ket.ExpectationPauliTerms_(ham.ham_);
    }
    return ket.ExpectationSparse_(ham);
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ_right,
                                            const circuit_t& circ_left,
                                            const parameter::ParameterResolver& pr) const -> py_qs_data_t {
    // Slow path: fallback to state-vector expectation for non-hermitian circuits.
    auto ket = derived_t(*this);
    auto bra = derived_t(*this);
    ket.ApplyCircuit(circ_right, pr);
    bra.ApplyCircuit(circ_left, pr);
    auto v_ket = ket.GetQS();
    auto v_bra = bra.GetQS();
    if (v_ket.size() != v_bra.size()) {
        throw std::runtime_error("mqmps: bra/ket dimension mismatch.");
    }
    if (ham.how_to_ == ORIGIN) {
        py_qs_data_t out{0, 0};
        for (const auto& [pwords, coeff] : ham.ham_) {
            // Apply Pauli string to ket (in-place temp) then dot with bra.
            VT<complex_t> tmp(v_ket.begin(), v_ket.end());
            complex_t phase{1, 0};
            for (const auto& [q, p] : pwords) {
                const size_t mask = static_cast<size_t>(1ULL) << static_cast<size_t>(q);
                if (p == 'Z') {
                    for (size_t i = 0; i < tmp.size(); ++i) {
                        if (i & mask) {
                            tmp[i] = -tmp[i];
                        }
                    }
                } else if (p == 'X') {
                    for (size_t i = 0; i < tmp.size(); ++i) {
                        if ((i & mask) == 0) {
                            std::swap(tmp[i], tmp[i | mask]);
                        }
                    }
                } else if (p == 'Y') {
                    // Y = i X Z (on computational basis), implement via X+phase.
                    phase *= complex_t{0, 1};
                    for (size_t i = 0; i < tmp.size(); ++i) {
                        if (i & mask) {
                            tmp[i] = -tmp[i];
                        }
                    }
                    for (size_t i = 0; i < tmp.size(); ++i) {
                        if ((i & mask) == 0) {
                            std::swap(tmp[i], tmp[i | mask]);
                        }
                    }
                } else {
                    throw std::runtime_error("mqmps: unsupported pauli in non-hermitian expectation.");
                }
            }
            py_qs_data_t dot{0, 0};
            for (size_t i = 0; i < tmp.size(); ++i) {
                dot += std::conj(v_bra[i]) * (phase * tmp[i]);
            }
            out += static_cast<calc_type>(coeff) * dot;
        }
        return out;
    }
    if (ham.how_to_ == BACKEND) {
        return sparse::ExpectationOfCsr<calc_type, calc_type>(ham.ham_sparse_main_, ham.ham_sparse_second_,
                                                              reinterpret_cast<calc_type*>(v_bra.data()),
                                                              reinterpret_cast<calc_type*>(v_ket.data()));
    }
    if (ham.how_to_ == FRONTEND) {
        return sparse::ExpectationOfCsr<calc_type, calc_type>(ham.ham_sparse_main_, reinterpret_cast<calc_type*>(v_bra.data()),
                                                              reinterpret_cast<calc_type*>(v_ket.data()));
    }
    throw std::runtime_error("mqmps: unsupported Hamiltonian type in non-hermitian expectation.");
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ_right,
                                            const circuit_t& circ_left, const derived_t& simulator_left,
                                            const parameter::ParameterResolver& pr) const -> py_qs_data_t {
    auto ket = derived_t(*this);
    auto bra = derived_t(simulator_left);
    ket.ApplyCircuit(circ_right, pr);
    bra.ApplyCircuit(circ_left, pr);
    auto v_ket = ket.GetQS();
    auto v_bra = bra.GetQS();
    if (ham.how_to_ == BACKEND) {
        return sparse::ExpectationOfCsr<calc_type, calc_type>(ham.ham_sparse_main_, ham.ham_sparse_second_,
                                                              reinterpret_cast<calc_type*>(v_bra.data()),
                                                              reinterpret_cast<calc_type*>(v_ket.data()));
    }
    if (ham.how_to_ == FRONTEND) {
        return sparse::ExpectationOfCsr<calc_type, calc_type>(ham.ham_sparse_main_, reinterpret_cast<calc_type*>(v_bra.data()),
                                                              reinterpret_cast<calc_type*>(v_ket.data()));
    }
    if (ham.how_to_ == ORIGIN) {
        return GetExpectation(ham, circ_right, circ_left, pr);
    }
    throw std::runtime_error("mqmps: unsupported Hamiltonian type.");
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::ApplyHamiltonian(const Hamiltonian<calc_type>& ham) {
    // Correctness-first fallback: apply in full state-vector form (not scalable).
    auto psi = GetQS();
    if (ham.how_to_ == ORIGIN) {
        // Build sparse Hamiltonian on demand.
        auto csr = sparse::SparseHamiltonian(ham.ham_, static_cast<Index>(n_qubits_));
        auto new_vec_ptr = sparse::Csr_Dot_Vec<calc_type, calc_type>(csr, reinterpret_cast<calc_type*>(psi.data()));
        auto new_vec_c = reinterpret_cast<complex_t*>(new_vec_ptr);
        VT<complex_t> new_vec(static_cast<size_t>(psi.size()));
        std::copy(new_vec_c, new_vec_c + psi.size(), new_vec.begin());
        free(new_vec_ptr);
        SetFromStateVectorLogical_(new_vec);
        return;
    }
    if (ham.how_to_ == BACKEND) {
        auto new_vec_ptr = sparse::Csr_Dot_Vec<calc_type, calc_type>(ham.ham_sparse_main_, ham.ham_sparse_second_,
                                                                     reinterpret_cast<calc_type*>(psi.data()));
        auto new_vec_c = reinterpret_cast<complex_t*>(new_vec_ptr);
        VT<complex_t> new_vec(static_cast<size_t>(psi.size()));
        std::copy(new_vec_c, new_vec_c + psi.size(), new_vec.begin());
        free(new_vec_ptr);
        SetFromStateVectorLogical_(new_vec);
        return;
    }
    if (ham.how_to_ == FRONTEND) {
        auto new_vec_ptr =
            sparse::Csr_Dot_Vec<calc_type, calc_type>(ham.ham_sparse_main_, reinterpret_cast<calc_type*>(psi.data()));
        auto new_vec_c = reinterpret_cast<complex_t*>(new_vec_ptr);
        VT<complex_t> new_vec(static_cast<size_t>(psi.size()));
        std::copy(new_vec_c, new_vec_c + psi.size(), new_vec.begin());
        free(new_vec_ptr);
        SetFromStateVectorLogical_(new_vec);
        return;
    }
    throw std::runtime_error("mqmps: unsupported Hamiltonian type for ApplyHamiltonian.");
}

template <typename qs_policy_t_>
VVT<typename MPSState<qs_policy_t_>::py_qs_data_t> MPSState<qs_policy_t_>::GetCircuitMatrix(
    const circuit_t& circ, const parameter::ParameterResolver& pr) const {
    (void)circ;
    (void)pr;
    throw std::runtime_error("mqmps: get_circuit_matrix is not supported yet.");
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::Purity() const -> calc_type {
    return static_cast<calc_type>(1);
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::PureStateVector() const -> VT<py_qs_data_t> {
    return GetQS();
}

template <typename qs_policy_t_>
template <typename policy_des>
auto MPSState<qs_policy_t_>::astype(unsigned new_seed) const -> MPSState<policy_des> {
    using out_t = MPSState<policy_des>;
    out_t out(n_qubits_, new_seed, max_bond_dimension_);
    auto v = GetQS();
    VT<typename out_t::py_qs_data_t> casted(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        casted[i] = static_cast<typename out_t::py_qs_data_t>(v[i]);
    }
    out.SetQS(casted);
    return out;
}
}  // namespace mindquantum::sim::mps::detail

#endif  // INCLUDE_SIMULATOR_MPS_MPS_STATE_TPP
