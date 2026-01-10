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
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

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

template <typename T>
struct TwoSiteWorkspace {
    using MatT = Mat<T>;
    MatT L;
    MatT R;
    MatT Theta;
    Eigen::BDCSVD<MatT> svd;
    Eigen::Index svd_rows_cap = 0;
    Eigen::Index svd_cols_cap = 0;
    unsigned int svd_opts = 0;

    void EnsureCapacity(size_t bond_dim_cap) {
        // Allocate for worst-case local sizes:
        //   L: (2*D_L) x D_M, R: D_M x (2*D_R), Theta: (2*D_L) x (2*D_R)
        // with D_L,D_M,D_R <= bond_dim_cap.
        const auto cap = static_cast<Eigen::Index>(bond_dim_cap);
        if (cap <= 0) {
            return;
        }
        const Eigen::Index two_cap = 2 * cap;

        if (L.size() < two_cap * cap) {
            L.resize(two_cap, cap);
        }
        if (R.size() < cap * two_cap) {
            R.resize(cap, two_cap);
        }
        if (Theta.size() < two_cap * two_cap) {
            Theta.resize(two_cap, two_cap);
        }
    }

    void EnsureSvdCapacity(Eigen::Index rows, Eigen::Index cols, unsigned int opts) {
        if (rows <= svd_rows_cap && cols <= svd_cols_cap && opts == svd_opts) {
            return;
        }
        svd = Eigen::BDCSVD<MatT>(rows, cols, opts);
        svd_rows_cap = rows;
        svd_cols_cap = cols;
        svd_opts = opts;
    }

    void ComputeSvd(const MatT& a, unsigned int opts) {
        EnsureSvdCapacity(a.rows(), a.cols(), opts);
        svd.compute(a, opts);
    }
};
}  // namespace detail

namespace detail {
template <typename calc_type>
inline VT<PauliTerm<calc_type>> ToPhysicalMergedPauliTerms(const VT<PauliTerm<calc_type>>& terms,
                                                           const qbits_t& pos_of_logical, qbit_t n_qubits) {
    VT<PauliTerm<calc_type>> phys_terms;
    phys_terms.reserve(terms.size());

    for (const auto& [pwords, coeff_in] : terms) {
        const auto coeff = static_cast<calc_type>(coeff_in);
        if (coeff == static_cast<calc_type>(0)) {
            continue;
        }
        VT<PauliWord> phys_words;
        phys_words.reserve(pwords.size());
        for (const auto& [q_idx, p] : pwords) {
            const auto q = static_cast<size_t>(q_idx);
            if (q >= pos_of_logical.size()) {
                throw std::runtime_error("mqmps: invalid qubit index in Pauli term.");
            }
            const auto pos = pos_of_logical[q];
            if (pos < 0 || pos >= n_qubits) {
                throw std::runtime_error("mqmps: invalid physical position in Pauli term.");
            }
            if (p != 'X' && p != 'Y' && p != 'Z') {
                throw std::runtime_error("mqmps: unknown Pauli word.");
            }
            phys_words.push_back(PauliWord{static_cast<Index>(pos), p});
        }
        std::sort(phys_words.begin(), phys_words.end(), [](const PauliWord& a, const PauliWord& b) {
            return (a.first != b.first) ? (a.first < b.first) : (a.second < b.second);
        });
        for (size_t i = 1; i < phys_words.size(); ++i) {
            if (phys_words[i].first == phys_words[i - 1].first) {
                throw std::runtime_error("mqmps: Pauli term has repeated qubit index.");
            }
        }
        phys_terms.push_back(PauliTerm<calc_type>{std::move(phys_words), coeff});
    }

    std::sort(phys_terms.begin(), phys_terms.end(), [](const PauliTerm<calc_type>& a, const PauliTerm<calc_type>& b) {
        return std::lexicographical_compare(a.first.begin(), a.first.end(), b.first.begin(), b.first.end());
    });

    VT<PauliTerm<calc_type>> merged;
    merged.reserve(phys_terms.size());
    for (auto& t : phys_terms) {
        if (merged.empty() || merged.back().first != t.first) {
            merged.push_back(std::move(t));
        } else {
            merged.back().second += t.second;
        }
    }
    merged.erase(std::remove_if(merged.begin(), merged.end(),
                                [](const PauliTerm<calc_type>& t) { return t.second == static_cast<calc_type>(0); }),
                 merged.end());
    return merged;
}

constexpr size_t kKiB = 1024ULL;
constexpr size_t kMiB = 1024ULL * kKiB;
constexpr size_t kGiB = 1024ULL * kMiB;

// Default safety caps for APIs that materialize dense objects (statevectors / density matrices).
// These operations can easily allocate multiple GiB and are not typical hot paths.
constexpr size_t kMaxDenseStateVectorOutputBytes = 1ULL * kGiB;
constexpr size_t kMaxReducedDensityMatrixOutputBytes = 1ULL * kGiB;

template <typename complex_t, typename calc_type, typename ApplyLocal, typename GetOp, typename IdentityValue>
inline complex_t EvaluateMergedPauliSumWithPrefixReuse(qbit_t n_qubits, const VT<PauliTerm<calc_type>>& merged_terms,
                                                       const std::array<complex_t, 4>& eye, ApplyLocal&& apply_local,
                                                       GetOp&& get_op, IdentityValue&& identity_value,
                                                       bool check_scalar) {
    using MatT = Mat<complex_t>;

    complex_t out{0, 0};
    VT<PauliWord> prev_ops;
    VT<MatT> prefix_env;
    VT<qbit_t> prefix_next_site;
    MatT e(1, 1);
    MatT e_tmp(1, 1);

    for (const auto& [ops, coeff] : merged_terms) {
        if (coeff == static_cast<calc_type>(0)) {
            continue;
        }
        if (ops.empty()) {
            out += coeff * identity_value();
            prev_ops.clear();
            prefix_env.clear();
            prefix_next_site.clear();
            continue;
        }

        size_t lcp = 0;
        while (lcp < prev_ops.size() && lcp < ops.size() && prev_ops[lcp] == ops[lcp]) {
            ++lcp;
        }
        prefix_env.resize(lcp);
        prefix_next_site.resize(lcp);

        qbit_t next_site = 0;
        e.resize(1, 1);
        e(0, 0) = complex_t{1, 0};
        if (lcp > 0) {
            e = prefix_env[lcp - 1];
            next_site = prefix_next_site[lcp - 1];
        }

        // Apply Pauli ops, interleaving identity sites.
        for (size_t op_idx = lcp; op_idx < ops.size(); ++op_idx) {
            const auto& op = ops[op_idx];
            const auto op_pos = static_cast<qbit_t>(op.first);
            if (op_pos < next_site) {
                throw std::runtime_error("mqmps: invalid Pauli op ordering.");
            }
            for (qbit_t s = next_site; s < op_pos; ++s) {
                apply_local(s, eye, e, &e_tmp);
                e.swap(e_tmp);
            }
            apply_local(op_pos, get_op(op.second), e, &e_tmp);
            e.swap(e_tmp);
            next_site = op_pos + 1;

            prefix_env.push_back(e);
            prefix_next_site.push_back(next_site);
        }

        // Tail identities to the end.
        for (qbit_t s = next_site; s < n_qubits; ++s) {
            apply_local(s, eye, e, &e_tmp);
            e.swap(e_tmp);
        }
        if (check_scalar && (e.rows() != 1 || e.cols() != 1)) {
            throw std::runtime_error("mqmps: expectation did not contract to scalar.");
        }
        out += coeff * e(0, 0);
        prev_ops = ops;
    }
    return out;
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
size_t MPSState<qs_policy_t_>::BondDimReserve_(qbit_t bond) const {
    if (bond < 0) {
        return 1;
    }
    const auto n = static_cast<size_t>(n_qubits_);
    const auto i = static_cast<size_t>(bond);
    if (i > n) {
        return 1;
    }
    const size_t exp = std::min(i, n - i);
    size_t max_no_cap = 0;
    if (exp >= (sizeof(size_t) * 8U - 1U)) {
        max_no_cap = std::numeric_limits<size_t>::max();
    } else {
        max_no_cap = static_cast<size_t>(1ULL) << exp;
    }

    if (max_bond_dimension_ != 0) {
        return std::min<size_t>(static_cast<size_t>(max_bond_dimension_), max_no_cap);
    }
    // Uncapped growth: reserve a small base to reduce churn without exploding memory use.
    constexpr size_t kDefaultReserveBondDim = 64;
    return std::min<size_t>(kDefaultReserveBondDim, max_no_cap);
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::ReserveStorage_() {
    const auto n = static_cast<size_t>(n_qubits_);
    if (lambdas_.size() != n + 1) {
        lambdas_.resize(n + 1);
    }
    if (gammas_.size() != n) {
        gammas_.resize(n);
    }

    for (qbit_t i = 0; i <= n_qubits_; ++i) {
        lambdas_.at(static_cast<size_t>(i)).reserve(BondDimReserve_(i));
    }
    for (qbit_t i = 0; i < n_qubits_; ++i) {
        const size_t dl_cap = BondDimReserve_(i);
        const size_t dr_cap = BondDimReserve_(i + 1);
        size_t site_cap = 0;
        if (dl_cap != 0 && dr_cap > std::numeric_limits<size_t>::max() / dl_cap) {
            site_cap = std::numeric_limits<size_t>::max();
        } else {
            site_cap = dl_cap * dr_cap;
        }
        gammas_.at(static_cast<size_t>(i)).g0.reserve(site_cap);
        gammas_.at(static_cast<size_t>(i)).g1.reserve(site_cap);
    }
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::InitZeroState_() {
    ReserveStorage_();
    // Initialize all Λ bonds to [1] while preserving capacity.
    for (qbit_t i = 0; i <= n_qubits_; ++i) {
        auto& lam = lambdas_.at(static_cast<size_t>(i));
        lam.resize(1);
        lam[0] = static_cast<calc_type>(1);
    }
    // Initialize all Γ sites to |0> while preserving capacity.
    for (qbit_t i = 0; i < n_qubits_; ++i) {
        auto& g = gammas_.at(static_cast<size_t>(i));
        g.dl = 1;
        g.dr = 1;
        g.g0.resize(1);
        g.g1.resize(1);
        g.g0[0] = complex_t{1, 0};
        g.g1[0] = complex_t{0, 0};
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
    const auto n = static_cast<size_t>(n_qubits_);
    if (n >= std::numeric_limits<size_t>::digits) {
        throw std::runtime_error("mqmps: GetQS() constructs a full 2^n statevector; requested n is too large.");
    }
    const size_t dim = size_t{1} << n;
    if (dim > detail::kMaxDenseStateVectorOutputBytes / sizeof(complex_t)) {
        throw std::runtime_error(
            "mqmps: GetQS() constructs a full 2^n statevector and is limited by max output bytes; n=" +
            std::to_string(n_qubits_) + ", elements=" + std::to_string(dim) +
            ", bytes_per_element=" + std::to_string(sizeof(complex_t)) +
            ", limit=" + std::to_string(detail::kMaxDenseStateVectorOutputBytes) + " bytes.");
    }
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

    // Ensure hot-path reserves are in place after rebuilding from a full vector.
    ReserveStorage_();
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::SetFromStateVectorLogical_(const VT<complex_t>& vec) {
    // Reset internal mapping and treat vec as physical ordering.
    SetFromStateVectorPhysical_(vec);
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::SetQS(const VT<py_qs_data_t>& qs) {
    const auto n = static_cast<size_t>(n_qubits_);
    if (n >= std::numeric_limits<size_t>::digits) {
        throw std::runtime_error("mqmps: SetQS() requires a full 2^n statevector; requested n is too large.");
    }
    const size_t dim = size_t{1} << n;
    if (qs.size() != dim) {
        throw std::runtime_error("mqmps: SetQS vector size mismatch.");
    }
    if (dim > detail::kMaxDenseStateVectorOutputBytes / sizeof(complex_t)) {
        throw std::runtime_error(
            "mqmps: SetQS() requires a full 2^n statevector and is limited by max output bytes; n=" +
            std::to_string(n_qubits_) + ", elements=" + std::to_string(dim) +
            ", bytes_per_element=" + std::to_string(sizeof(complex_t)) +
            ", limit=" + std::to_string(detail::kMaxDenseStateVectorOutputBytes) + " bytes.");
    }
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

    constexpr unsigned int svd_opts = Eigen::ComputeThinU | Eigen::ComputeThinV;
    thread_local detail::TwoSiteWorkspace<complex_t> ws;
    const size_t bond_dim_cap = [&]() {
        if (max_bond_dimension_ != 0) {
            return static_cast<size_t>(max_bond_dimension_);
        }
        constexpr size_t kDefaultWorkspaceBondDim = 64;
        return std::max({D_L, D_M, D_R, kDefaultWorkspaceBondDim});
    }();
    ws.EnsureCapacity(bond_dim_cap);

    auto& L = ws.L;
    auto& R = ws.R;
    auto& Theta = ws.Theta;
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

    ws.ComputeSvd(Theta, svd_opts);
    const auto& s_all = ws.svd.singularValues();
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
    const auto U = ws.svd.matrixU().leftCols(static_cast<Eigen::Index>(D_new));
    const auto V = ws.svd.matrixV().leftCols(static_cast<Eigen::Index>(D_new));  // (2*D_R x D_new)

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
void MPSState<qs_policy_t_>::ReorderToTargetMapping_(const qbits_t& logical_at_pos_target) {
    if (logical_at_pos_target.size() != static_cast<size_t>(n_qubits_)) {
        throw std::runtime_error("mqmps: invalid target mapping size.");
    }
    if (logical_at_pos_.size() != static_cast<size_t>(n_qubits_) || pos_of_logical_.size() != static_cast<size_t>(n_qubits_)) {
        throw std::runtime_error("mqmps: internal mapping size mismatch.");
    }
    if (logical_at_pos_ == logical_at_pos_target) {
        return;
    }

    // Validate target is a permutation of [0..n_qubits_-1].
    VT<int> seen(static_cast<size_t>(n_qubits_), 0);
    for (auto q : logical_at_pos_target) {
        if (q < 0 || q >= n_qubits_) {
            throw std::runtime_error("mqmps: target mapping contains out-of-range logical index.");
        }
        if (++seen.at(static_cast<size_t>(q)) != 1) {
            throw std::runtime_error("mqmps: target mapping contains duplicates.");
        }
    }

    const std::array<complex_t, 16> swap4 = {complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{0, 0},
                                             complex_t{0, 0}, complex_t{0, 0}, complex_t{1, 0}, complex_t{0, 0},
                                             complex_t{0, 0}, complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0},
                                             complex_t{0, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{1, 0}};

    const auto do_swap = [&](qbit_t bond) {
        Apply2QOnBond_(bond, swap4);
        const auto ql = logical_at_pos_.at(static_cast<size_t>(bond));
        const auto qr = logical_at_pos_.at(static_cast<size_t>(bond + 1));
        std::swap(logical_at_pos_.at(static_cast<size_t>(bond)), logical_at_pos_.at(static_cast<size_t>(bond + 1)));
        pos_of_logical_.at(static_cast<size_t>(ql)) = bond + 1;
        pos_of_logical_.at(static_cast<size_t>(qr)) = bond;
    };

    // Deterministic 1D swap routing to the target permutation.
    for (qbit_t pos = 0; pos < n_qubits_; ++pos) {
        const auto desired = logical_at_pos_target.at(static_cast<size_t>(pos));
        qbit_t cur_pos = pos_of_logical_.at(static_cast<size_t>(desired));
        while (cur_pos > pos) {
            do_swap(cur_pos - 1);
            cur_pos = pos_of_logical_.at(static_cast<size_t>(desired));
        }
        while (cur_pos < pos) {
            do_swap(cur_pos);
            cur_pos = pos_of_logical_.at(static_cast<size_t>(desired));
        }
    }

    if (logical_at_pos_ != logical_at_pos_target) {
        throw std::runtime_error("mqmps: failed to reach target mapping.");
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

            // One-shot measurement of the final state (native MPS path):
            // - sample a full bitstring in physical order using the canonical Γ/Λ form,
            // - project measured sites in-place (no full state-vector construction),
            // - re-canonicalize via an identity sweep (also normalizes).
            using Mat = detail::Mat<complex_t>;
            using RowVec = detail::RowVec<complex_t>;

            // Sample full register bits in physical site order.
            VT<int> bits(static_cast<size_t>(n_qubits_), 0);
            if (n_qubits_ > 0) {
                RowVec x(1);
                x(0) = complex_t{1, 0};
                for (qbit_t i = 0; i < n_qubits_; ++i) {
                    const auto D_i = lambdas_.at(static_cast<size_t>(i)).size();
                    const auto D_next = lambdas_.at(static_cast<size_t>(i + 1)).size();
                    const auto& lam_next = lambdas_.at(static_cast<size_t>(i + 1));
                    const auto& site = gammas_.at(static_cast<size_t>(i));
                    if (site.dl != D_i || site.dr != D_next) {
                        throw std::runtime_error("mqmps: inconsistent dimensions in measurement sampling.");
                    }
                    Eigen::Map<const Mat> g0(site.g0.data(), static_cast<Eigen::Index>(D_i),
                                             static_cast<Eigen::Index>(D_next));
                    Eigen::Map<const Mat> g1(site.g1.data(), static_cast<Eigen::Index>(D_i),
                                             static_cast<Eigen::Index>(D_next));
                    if (x.cols() != static_cast<Eigen::Index>(D_i)) {
                        throw std::runtime_error("mqmps: internal measurement sampling dim mismatch.");
                    }
                    RowVec y0 = x * g0;
                    RowVec y1 = x * g1;
                    for (size_t k = 0; k < D_next; ++k) {
                        y0(static_cast<Eigen::Index>(k)) *= lam_next[k];
                        y1(static_cast<Eigen::Index>(k)) *= lam_next[k];
                    }

                    const double p0 = static_cast<double>(y0.squaredNorm());
                    const double p1 = static_cast<double>(y1.squaredNorm());
                    const double denom = p0 + p1;
                    if (denom <= 0) {
                        throw std::runtime_error("mqmps: invalid probability in measurement sampling.");
                    }
                    const double prob0 = p0 / denom;
                    const bool bit0 = (rng_() < prob0);
                    bits.at(static_cast<size_t>(i)) = bit0 ? 0 : 1;
                    x = bit0 ? y0 : y1;
                }
            }

            struct MeasInfo {
                qbit_t pos = 0;  // physical site index
                int outcome = 0;
                bool do_reset = false;
                index_t reset_to = 0;
            };
            VT<MeasInfo> meas_ops;
            meas_ops.reserve(stage.meas.size());
            for (const auto& m : stage.meas) {
                if (m.pos < 0 || m.pos >= n_qubits_) {
                    throw std::runtime_error("mqmps: invalid measurement site index.");
                }
                const int bit = bits.at(static_cast<size_t>(m.pos));
                meas_result[m.key] = bit;
                meas_ops.push_back(MeasInfo{m.pos, bit, m.reset, m.reset_to});
            }

            // Collapse: apply Z-basis projectors at measured physical sites (unnormalized).
            for (const auto& info : meas_ops) {
                auto& site = gammas_.at(static_cast<size_t>(info.pos));
                if (info.outcome == 0) {
                    std::fill(site.g1.begin(), site.g1.end(), complex_t{0, 0});
                } else {
                    std::fill(site.g0.begin(), site.g0.end(), complex_t{0, 0});
                }
            }

            // Apply post-measure reset as a physical X (swap Γ0/Γ1).
            for (const auto& info : meas_ops) {
                if (!info.do_reset) {
                    continue;
                }
                if (static_cast<index_t>(info.outcome) == info.reset_to) {
                    continue;
                }
                auto& site = gammas_.at(static_cast<size_t>(info.pos));
                std::swap(site.g0, site.g1);
            }

            // Re-canonicalize + normalize by sweeping an identity 2Q update across the chain.
            if (n_qubits_ == 1) {
                auto& site = gammas_.at(0);
                calc_type norm2 = 0;
                for (const auto& v : site.g0) {
                    norm2 += static_cast<calc_type>(std::norm(v));
                }
                for (const auto& v : site.g1) {
                    norm2 += static_cast<calc_type>(std::norm(v));
                }
                if (norm2 == static_cast<calc_type>(0)) {
                    throw std::runtime_error("mqmps: collapse produced zero state.");
                }
                const calc_type inv_norm = static_cast<calc_type>(1) / std::sqrt(norm2);
                for (auto& v : site.g0) {
                    v *= inv_norm;
                }
                for (auto& v : site.g1) {
                    v *= inv_norm;
                }
            } else if (n_qubits_ > 1) {
                const std::array<complex_t, 16> eye4 = {complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0},
                                                        complex_t{0, 0}, complex_t{0, 0}, complex_t{1, 0},
                                                        complex_t{0, 0}, complex_t{0, 0}, complex_t{0, 0},
                                                        complex_t{0, 0}, complex_t{1, 0}, complex_t{0, 0},
                                                        complex_t{0, 0}, complex_t{0, 0}, complex_t{0, 0},
                                                        complex_t{1, 0}};
                for (qbit_t bond = 0; bond < n_qubits_ - 1; ++bond) {
                    Apply2QOnBond_(bond, eye4);
                }
            }
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
            const double p0r = static_cast<double>(std::real(p0));
            const double p1r = static_cast<double>(std::real(p1));
            if (p0r < 0 || p1r < 0) {
                throw std::runtime_error("mqmps: invalid probability in sampling.");
            }
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
VT<typename MPSState<qs_policy_t_>::py_qs_data_t> MPSState<qs_policy_t_>::GetReducedDensityMatrix(
    const qbits_t& kept_qubits) const {
    using Mat = detail::Mat<complex_t>;
    const size_t n = static_cast<size_t>(n_qubits_);
    const size_t k_in = kept_qubits.size();
    if (k_in == 0 || k_in >= n) {
        throw std::runtime_error("mqmps: kept_qubits must be between 1 and n_qubits-1.");
    }

    // Normalize input: unique + sorted logical qubit indices.
    VT<qbit_t> kept_logical = kept_qubits;
    std::sort(kept_logical.begin(), kept_logical.end());
    kept_logical.erase(std::unique(kept_logical.begin(), kept_logical.end()), kept_logical.end());
    if (kept_logical.size() != k_in) {
        throw std::runtime_error("mqmps: kept_qubits contains duplicates.");
    }
    const size_t k = kept_logical.size();
    if (k >= std::numeric_limits<size_t>::digits) {
        throw std::runtime_error("mqmps: kept_qubits is too large to build a reduced density matrix.");
    }
    const size_t dim_keep = size_t{1} << k;
    if (dim_keep > std::numeric_limits<size_t>::max() / dim_keep) {
        throw std::runtime_error("mqmps: kept_qubits is too large to build a reduced density matrix.");
    }
    const size_t dim_rho = dim_keep * dim_keep;
    if (dim_rho > detail::kMaxReducedDensityMatrixOutputBytes / sizeof(complex_t)) {
        throw std::runtime_error(
            "mqmps: reduced density matrix is limited by max output bytes; kept_qubits=" + std::to_string(k) +
            ", elements=" + std::to_string(dim_rho) + ", bytes_per_element=" + std::to_string(sizeof(complex_t)) +
            ", limit=" + std::to_string(detail::kMaxReducedDensityMatrixOutputBytes) + " bytes.");
    }

    // Map logical kept qubits -> physical sites (dynamic mapping).
    // We contract in physical order (MPS sites), but output indices are in logical order (sorted qubit indices).
    struct KeepStep {
        qbit_t pos = 0;
        size_t log_bit = 0;  // bit index in the reduced density matrix basis (little-endian by logical qubit index)
    };
    VT<KeepStep> keep_steps;
    keep_steps.reserve(k);
    for (size_t log_bit = 0; log_bit < k; ++log_bit) {
        const auto q = kept_logical.at(log_bit);
        if (q < 0 || q >= n_qubits_) {
            throw std::runtime_error("mqmps: invalid kept qubit index.");
        }
        const auto pos = pos_of_logical_.at(static_cast<size_t>(q));
        if (pos < 0 || pos >= n_qubits_) {
            throw std::runtime_error("mqmps: invalid physical position for kept qubit.");
        }
        keep_steps.push_back(KeepStep{pos, log_bit});
    }
    std::sort(keep_steps.begin(), keep_steps.end(), [](const KeepStep& a, const KeepStep& b) { return a.pos < b.pos; });
    for (size_t j = 1; j < keep_steps.size(); ++j) {
        if (keep_steps[j].pos == keep_steps[j - 1].pos) {
            throw std::runtime_error("mqmps: internal error: two logical qubits mapped to the same physical site.");
        }
    }

    VT<complex_t> out(dim_rho);

    const auto scale_by_lambda = [](const VT<calc_type>& lam_next, Mat* m) {
        const size_t D = lam_next.size();
        for (size_t t = 0; t < D; ++t) {
            m->row(static_cast<Eigen::Index>(t)) *= lam_next[t];
            m->col(static_cast<Eigen::Index>(t)) *= lam_next[t];
        }
    };

    const auto apply_traced_site = [&](qbit_t site, Mat* e, Mat* a0, Mat* a1, Mat* tmp) {
        const size_t D_i = lambdas_[static_cast<size_t>(site)].size();
        const size_t D_next = lambdas_[static_cast<size_t>(site + 1)].size();
        const auto& lam_next = lambdas_[static_cast<size_t>(site + 1)];
        const auto& g = gammas_[static_cast<size_t>(site)];
        if (g.dl != D_i || g.dr != D_next) {
            throw std::runtime_error("mqmps: inconsistent gamma/lambda dimensions during reduced density matrix.");
        }
        if (static_cast<size_t>(e->rows()) != D_i || static_cast<size_t>(e->cols()) != D_i) {
            throw std::runtime_error("mqmps: internal env dim mismatch in reduced density matrix.");
        }

        Eigen::Map<const Mat> g0(g.g0.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        Eigen::Map<const Mat> g1(g.g1.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));

        a0->resize(static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        a1->resize(static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        a0->noalias() = (*e) * g0;
        a1->noalias() = (*e) * g1;

        tmp->resize(static_cast<Eigen::Index>(D_next), static_cast<Eigen::Index>(D_next));
        tmp->noalias() = g0.adjoint() * (*a0);
        tmp->noalias() += g1.adjoint() * (*a1);
        scale_by_lambda(lam_next, tmp);
        e->swap(*tmp);
    };

    const auto contract_traced_range = [&](qbit_t begin_site, qbit_t end_site, Mat* e, Mat* a0, Mat* a1, Mat* tmp) {
        if (end_site < begin_site) {
            throw std::runtime_error("mqmps: internal error contracting reduced density matrix (invalid site range).");
        }
        for (qbit_t s = begin_site; s < end_site; ++s) {
            apply_traced_site(s, e, a0, a1, tmp);
        }
    };

    struct RdmWs {
        // Temporary buffers for transfer contractions.
        Mat a0;
        Mat a1;
        Mat tmp;

        // Per-level storage for a DFS path within one task.
        VT<Mat> env_stack;
        VT<Mat> ket0_stack;
        VT<Mat> ket1_stack;
    };

    const auto dfs_serial = [&](auto&& self, size_t depth, qbit_t next_site, size_t row_acc, size_t col_acc,
                                RdmWs* ws) -> void {
        if (depth == k) {
            Mat& e = ws->env_stack[depth];
            contract_traced_range(next_site, n_qubits_, &e, &ws->a0, &ws->a1, &ws->tmp);
            if (e.rows() != 1 || e.cols() != 1) {
                throw std::runtime_error("mqmps: reduced density matrix did not contract to scalar.");
            }
            out[row_acc * dim_keep + col_acc] = e(0, 0);
            return;
        }

        Mat& e = ws->env_stack[depth];
        const qbit_t kept_pos = keep_steps[depth].pos;
        const size_t log_bit = keep_steps[depth].log_bit;
        if (kept_pos < next_site) {
            throw std::runtime_error("mqmps: internal error contracting reduced density matrix (kept site ordering).");
        }
        contract_traced_range(next_site, kept_pos, &e, &ws->a0, &ws->a1, &ws->tmp);

        const size_t D_i = lambdas_[static_cast<size_t>(kept_pos)].size();
        const size_t D_next = lambdas_[static_cast<size_t>(kept_pos + 1)].size();
        const auto& lam_next = lambdas_[static_cast<size_t>(kept_pos + 1)];
        const auto& g = gammas_[static_cast<size_t>(kept_pos)];
        if (g.dl != D_i || g.dr != D_next) {
            throw std::runtime_error("mqmps: inconsistent gamma/lambda dimensions during reduced density matrix.");
        }
        if (static_cast<size_t>(e.rows()) != D_i || static_cast<size_t>(e.cols()) != D_i) {
            throw std::runtime_error("mqmps: internal env dim mismatch in reduced density matrix.");
        }

        Eigen::Map<const Mat> g0(g.g0.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        Eigen::Map<const Mat> g1(g.g1.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));

        Mat& ek0 = ws->ket0_stack[depth];
        Mat& ek1 = ws->ket1_stack[depth];
        ek0.resize(static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        ek1.resize(static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        ek0.noalias() = e * g0;
        ek1.noalias() = e * g1;

        for (size_t ket_bit = 0; ket_bit < 2; ++ket_bit) {
            const Mat& tmp_k = (ket_bit == 0) ? ek0 : ek1;
            const size_t next_row = row_acc | (ket_bit << log_bit);
            for (size_t bra_bit = 0; bra_bit < 2; ++bra_bit) {
                const auto& gl = (bra_bit == 0) ? g0 : g1;
                const size_t next_col = col_acc | (bra_bit << log_bit);

                Mat& child = ws->env_stack[depth + 1];
                child.resize(static_cast<Eigen::Index>(D_next), static_cast<Eigen::Index>(D_next));
                child.noalias() = gl.adjoint() * tmp_k;
                scale_by_lambda(lam_next, &child);

                self(self, depth + 1, kept_pos + 1, next_row, next_col, ws);
            }
        }
    };

    // Parallelize only the top levels: each task owns its workspace and computes a full subtree serially.
    constexpr size_t kParallelKeptQubits = 3;  // 4^3 = 64 tasks.
    const size_t parallel_depth = std::min(kParallelKeptQubits, k);

    std::function<void(size_t, qbit_t, Mat, size_t, size_t)> dfs_parallel;
    dfs_parallel = [&](size_t depth, qbit_t next_site, Mat env, size_t row_acc, size_t col_acc) {
        if (depth == parallel_depth) {
            RdmWs ws;
            ws.env_stack.resize(k + 1);
            ws.ket0_stack.resize(k);
            ws.ket1_stack.resize(k);
            ws.env_stack[depth].swap(env);
            dfs_serial(dfs_serial, depth, next_site, row_acc, col_acc, &ws);
            return;
        }

        const qbit_t kept_pos = keep_steps[depth].pos;
        const size_t log_bit = keep_steps[depth].log_bit;
        if (kept_pos < next_site) {
            throw std::runtime_error("mqmps: internal error contracting reduced density matrix (kept site ordering).");
        }

        Mat a0;
        Mat a1;
        Mat tmp;
        contract_traced_range(next_site, kept_pos, &env, &a0, &a1, &tmp);

        const size_t D_i = lambdas_[static_cast<size_t>(kept_pos)].size();
        const size_t D_next = lambdas_[static_cast<size_t>(kept_pos + 1)].size();
        const auto& lam_next = lambdas_[static_cast<size_t>(kept_pos + 1)];
        const auto& g = gammas_[static_cast<size_t>(kept_pos)];
        if (g.dl != D_i || g.dr != D_next) {
            throw std::runtime_error("mqmps: inconsistent gamma/lambda dimensions during reduced density matrix.");
        }
        if (static_cast<size_t>(env.rows()) != D_i || static_cast<size_t>(env.cols()) != D_i) {
            throw std::runtime_error("mqmps: internal env dim mismatch in reduced density matrix.");
        }

        Eigen::Map<const Mat> g0(g.g0.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        Eigen::Map<const Mat> g1(g.g1.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));

        // Reuse a0/a1 as env*g0/env*g1 at the kept site.
        a0.resize(static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        a1.resize(static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        a0.noalias() = env * g0;
        a1.noalias() = env * g1;

        const size_t next_depth = depth + 1;
        const qbit_t next_site_child = kept_pos + 1;
        for (size_t ket_bit = 0; ket_bit < 2; ++ket_bit) {
            const Mat& tmp_k = (ket_bit == 0) ? a0 : a1;
            const size_t next_row = row_acc | (ket_bit << log_bit);
            for (size_t bra_bit = 0; bra_bit < 2; ++bra_bit) {
                const auto& gl = (bra_bit == 0) ? g0 : g1;
                const size_t next_col = col_acc | (bra_bit << log_bit);

                Mat child(static_cast<Eigen::Index>(D_next), static_cast<Eigen::Index>(D_next));
                child.noalias() = gl.adjoint() * tmp_k;
                scale_by_lambda(lam_next, &child);

#pragma omp task firstprivate(child, next_depth, next_site_child, next_row, next_col)
                { dfs_parallel(next_depth, next_site_child, std::move(child), next_row, next_col); }
            }
        }
    };

    Mat init_env(1, 1);
    init_env(0, 0) = complex_t{1, 0};
#pragma omp parallel
    {
#pragma omp single
        {
#pragma omp taskgroup
            { dfs_parallel(0, 0, std::move(init_env), 0, 0); }
        }
    }

    return out;
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::ExpectationPauliTerms_(const VT<PauliTerm<calc_type>>& terms) const -> py_qs_data_t {
    using Mat = detail::Mat<complex_t>;
    constexpr complex_t I{0, 1};

    if (!pauli_sum_cache_ || pauli_sum_cache_->pos_of_logical != pos_of_logical_ || pauli_sum_cache_->terms != terms) {
        PauliSumCache cache;
        cache.pos_of_logical = pos_of_logical_;
        cache.terms = terms;
        cache.phys_terms = detail::ToPhysicalMergedPauliTerms(terms, pos_of_logical_, n_qubits_);
        pauli_sum_cache_ = std::move(cache);
    }

    struct ExpectWs {
        Mat a0;
        Mat a1;
        Mat tmp;
    };
    thread_local ExpectWs ws;

    const std::array<complex_t, 4> eye = {complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{1, 0}};
    const std::array<complex_t, 4> xop = {complex_t{0, 0}, complex_t{1, 0}, complex_t{1, 0}, complex_t{0, 0}};
    const std::array<complex_t, 4> yop = {complex_t{0, 0}, -I, I, complex_t{0, 0}};
    const std::array<complex_t, 4> zop = {complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{-1, 0}};

    const auto get_pauli_op = [&](char p) -> const std::array<complex_t, 4>& {
        switch (p) {
            case 'X':
                return xop;
            case 'Y':
                return yop;
            case 'Z':
                return zop;
            default:
                throw std::runtime_error("mqmps: unknown Pauli word.");
        }
    };

    auto apply_local = [&](qbit_t site, const std::array<complex_t, 4>& op, const Mat& e, Mat* out) {
        const size_t D_i = lambdas_.at(static_cast<size_t>(site)).size();
        const size_t D_next = lambdas_.at(static_cast<size_t>(site + 1)).size();
        const auto& lam_next = lambdas_.at(static_cast<size_t>(site + 1));
        const auto& g = gammas_.at(static_cast<size_t>(site));
        if (g.dl != D_i || g.dr != D_next) {
            throw std::runtime_error("mqmps: inconsistent gamma/lambda dimensions during expectation.");
        }

        Eigen::Map<const Mat> g0(g.g0.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        Eigen::Map<const Mat> g1(g.g1.data(), static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));

        ws.a0.resize(static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        ws.a1.resize(static_cast<Eigen::Index>(D_i), static_cast<Eigen::Index>(D_next));
        ws.a0.noalias() = e * g0;
        ws.a1.noalias() = e * g1;

        out->resize(static_cast<Eigen::Index>(D_next), static_cast<Eigen::Index>(D_next));
        ws.tmp.resize(static_cast<Eigen::Index>(D_next), static_cast<Eigen::Index>(D_next));

        // out = o00*(g0^H a0) + o01*(g0^H a1) + o10*(g1^H a0) + o11*(g1^H a1)
        ws.tmp.noalias() = g0.adjoint() * ws.a0;
        *out = op[0] * ws.tmp;
        if (op[1] != complex_t{0, 0}) {
            ws.tmp.noalias() = g0.adjoint() * ws.a1;
            *out += op[1] * ws.tmp;
        }
        if (op[2] != complex_t{0, 0}) {
            ws.tmp.noalias() = g1.adjoint() * ws.a0;
            *out += op[2] * ws.tmp;
        }
        ws.tmp.noalias() = g1.adjoint() * ws.a1;
        *out += op[3] * ws.tmp;

        // out = diag(lam_next) * out * diag(lam_next)
        for (size_t k = 0; k < D_next; ++k) {
            out->row(static_cast<Eigen::Index>(k)) *= lam_next[k];
            out->col(static_cast<Eigen::Index>(k)) *= lam_next[k];
        }
    };

    return detail::EvaluateMergedPauliSumWithPrefixReuse<complex_t, calc_type>(
        n_qubits_, pauli_sum_cache_->phys_terms, eye, apply_local, get_pauli_op,
        []() { return complex_t{1, 0}; }, false);
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::ExpectationPauliTermsBraKet_(const VT<PauliTerm<calc_type>>& terms, const derived_t& bra) const
    -> py_qs_data_t {
    using Mat = detail::Mat<complex_t>;
    constexpr complex_t I{0, 1};

    if (bra.n_qubits_ != n_qubits_) {
        throw std::runtime_error("mqmps: bra/ket qubit number mismatch.");
    }
    if (bra.pos_of_logical_ != pos_of_logical_ || bra.logical_at_pos_ != logical_at_pos_) {
        throw std::runtime_error("mqmps: bra/ket mapping mismatch (internal alignment required).");
    }

    const auto merged = detail::ToPhysicalMergedPauliTerms(terms, pos_of_logical_, n_qubits_);

    struct CrossExpectWs {
        Mat ek0;
        Mat ek1;
        Mat tmp;
    };
    thread_local CrossExpectWs ws;

    const std::array<complex_t, 4> eye = {complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{1, 0}};
    const std::array<complex_t, 4> xop = {complex_t{0, 0}, complex_t{1, 0}, complex_t{1, 0}, complex_t{0, 0}};
    const std::array<complex_t, 4> yop = {complex_t{0, 0}, -I, I, complex_t{0, 0}};
    const std::array<complex_t, 4> zop = {complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{-1, 0}};

    const auto get_pauli_op = [&](char p) -> const std::array<complex_t, 4>& {
        switch (p) {
            case 'X':
                return xop;
            case 'Y':
                return yop;
            case 'Z':
                return zop;
            default:
                throw std::runtime_error("mqmps: unknown Pauli word.");
        }
    };

    auto apply_local = [&](qbit_t site, const std::array<complex_t, 4>& op, const Mat& e_in, Mat* out) {
        const size_t D_b = bra.lambdas_.at(static_cast<size_t>(site)).size();
        const size_t D_b_next = bra.lambdas_.at(static_cast<size_t>(site + 1)).size();
        const auto& lam_b_next = bra.lambdas_.at(static_cast<size_t>(site + 1));
        const auto& gb = bra.gammas_.at(static_cast<size_t>(site));
        if (gb.dl != D_b || gb.dr != D_b_next) {
            throw std::runtime_error("mqmps: inconsistent bra gamma/lambda dimensions during expectation.");
        }

        const size_t D_k = lambdas_.at(static_cast<size_t>(site)).size();
        const size_t D_k_next = lambdas_.at(static_cast<size_t>(site + 1)).size();
        const auto& lam_k_next = lambdas_.at(static_cast<size_t>(site + 1));
        const auto& gk = gammas_.at(static_cast<size_t>(site));
        if (gk.dl != D_k || gk.dr != D_k_next) {
            throw std::runtime_error("mqmps: inconsistent ket gamma/lambda dimensions during expectation.");
        }

        if (static_cast<size_t>(e_in.rows()) != D_b || static_cast<size_t>(e_in.cols()) != D_k) {
            throw std::runtime_error("mqmps: internal bra/ket env dim mismatch during expectation.");
        }

        Eigen::Map<const Mat> g0b(gb.g0.data(), static_cast<Eigen::Index>(D_b), static_cast<Eigen::Index>(D_b_next));
        Eigen::Map<const Mat> g1b(gb.g1.data(), static_cast<Eigen::Index>(D_b), static_cast<Eigen::Index>(D_b_next));
        Eigen::Map<const Mat> g0k(gk.g0.data(), static_cast<Eigen::Index>(D_k), static_cast<Eigen::Index>(D_k_next));
        Eigen::Map<const Mat> g1k(gk.g1.data(), static_cast<Eigen::Index>(D_k), static_cast<Eigen::Index>(D_k_next));

        ws.ek0.resize(static_cast<Eigen::Index>(D_b), static_cast<Eigen::Index>(D_k_next));
        ws.ek1.resize(static_cast<Eigen::Index>(D_b), static_cast<Eigen::Index>(D_k_next));
        ws.ek0.noalias() = e_in * g0k;
        ws.ek1.noalias() = e_in * g1k;

        out->resize(static_cast<Eigen::Index>(D_b_next), static_cast<Eigen::Index>(D_k_next));
        out->setZero();
        ws.tmp.resize(static_cast<Eigen::Index>(D_b_next), static_cast<Eigen::Index>(D_k_next));

        if (op[0] != complex_t{0, 0}) {
            ws.tmp.noalias() = g0b.adjoint() * ws.ek0;
            out->noalias() += op[0] * ws.tmp;
        }
        if (op[1] != complex_t{0, 0}) {
            ws.tmp.noalias() = g0b.adjoint() * ws.ek1;
            out->noalias() += op[1] * ws.tmp;
        }
        if (op[2] != complex_t{0, 0}) {
            ws.tmp.noalias() = g1b.adjoint() * ws.ek0;
            out->noalias() += op[2] * ws.tmp;
        }
        if (op[3] != complex_t{0, 0}) {
            ws.tmp.noalias() = g1b.adjoint() * ws.ek1;
            out->noalias() += op[3] * ws.tmp;
        }

        // out = diag(lam_b_next) * out * diag(lam_k_next)
        for (size_t r = 0; r < D_b_next; ++r) {
            out->row(static_cast<Eigen::Index>(r)) *= lam_b_next[r];
        }
        for (size_t c = 0; c < D_k_next; ++c) {
            out->col(static_cast<Eigen::Index>(c)) *= lam_k_next[c];
        }
    };

    std::optional<complex_t> overlap;
    const auto inner_product = [&]() -> complex_t {
        if (overlap) {
            return *overlap;
        }
        Mat e(1, 1);
        Mat e_tmp(1, 1);
        e(0, 0) = complex_t{1, 0};
        for (qbit_t s = 0; s < n_qubits_; ++s) {
            apply_local(s, eye, e, &e_tmp);
            e.swap(e_tmp);
        }
        if (e.rows() != 1 || e.cols() != 1) {
            throw std::runtime_error("mqmps: inner product did not contract to scalar.");
        }
        overlap = e(0, 0);
        return *overlap;
    };

    return detail::EvaluateMergedPauliSumWithPrefixReuse<complex_t, calc_type>(
        n_qubits_, merged, eye, apply_local, get_pauli_op, inner_product, true);
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
    throw std::runtime_error("mqmps: sparse Hamiltonian expectation is not supported; provide Pauli terms (ORIGIN).");
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ_right,
                                            const circuit_t& circ_left,
                                            const parameter::ParameterResolver& pr) const -> py_qs_data_t {
    auto ket = derived_t(*this);
    auto bra = derived_t(*this);
    ket.ApplyCircuit(circ_right, pr);
    bra.ApplyCircuit(circ_left, pr);
    if (ham.how_to_ == ORIGIN) {
        if (bra.logical_at_pos_ != ket.logical_at_pos_) {
            bra.ReorderToTargetMapping_(ket.logical_at_pos_);
        }
        return ket.ExpectationPauliTermsBraKet_(ham.ham_, bra);
    }
    throw std::runtime_error("mqmps: sparse Hamiltonian expectation is not supported; provide Pauli terms (ORIGIN).");
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ_right,
                                            const circuit_t& circ_left, const derived_t& simulator_left,
                                            const parameter::ParameterResolver& pr) const -> py_qs_data_t {
    auto ket = derived_t(*this);
    auto bra = derived_t(simulator_left);
    ket.ApplyCircuit(circ_right, pr);
    bra.ApplyCircuit(circ_left, pr);
    if (ham.how_to_ == ORIGIN) {
        if (bra.logical_at_pos_ != ket.logical_at_pos_) {
            bra.ReorderToTargetMapping_(ket.logical_at_pos_);
        }
        return ket.ExpectationPauliTermsBraKet_(ham.ham_, bra);
    }
    throw std::runtime_error("mqmps: sparse Hamiltonian expectation is not supported; provide Pauli terms (ORIGIN).");
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::ApplyHamiltonian(const Hamiltonian<calc_type>& ham) {
    (void)ham;
    throw std::runtime_error("mqmps: apply_hamiltonian is not supported (native MPO/apply path not implemented).");
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

    out.pos_of_logical_ = pos_of_logical_;
    out.logical_at_pos_ = logical_at_pos_;

    out.lambdas_.resize(lambdas_.size());
    for (size_t i = 0; i < lambdas_.size(); ++i) {
        const auto& src = lambdas_.at(i);
        auto& dst = out.lambdas_.at(i);
        dst.resize(src.size());
        for (size_t j = 0; j < src.size(); ++j) {
            dst.at(j) = static_cast<typename out_t::calc_type>(src.at(j));
        }
    }

    out.gammas_.resize(gammas_.size());
    for (size_t i = 0; i < gammas_.size(); ++i) {
        const auto& src = gammas_.at(i);
        auto& dst = out.gammas_.at(i);
        dst.dl = src.dl;
        dst.dr = src.dr;
        dst.g0.resize(src.g0.size());
        dst.g1.resize(src.g1.size());
        for (size_t j = 0; j < src.g0.size(); ++j) {
            dst.g0.at(j) = static_cast<typename out_t::complex_t>(src.g0.at(j));
        }
        for (size_t j = 0; j < src.g1.size(); ++j) {
            dst.g1.at(j) = static_cast<typename out_t::complex_t>(src.g1.at(j));
        }
    }

    out.pauli_sum_cache_.reset();
    out.ReserveStorage_();
    return out;
}
}  // namespace mindquantum::sim::mps::detail

#endif  // INCLUDE_SIMULATOR_MPS_MPS_STATE_TPP
