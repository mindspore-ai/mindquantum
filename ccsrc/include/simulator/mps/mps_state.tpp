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

#include <cmath>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/utils.h"
#include "simulator/mps/detail/linalg/blas.h"
#include "simulator/mps/detail/linalg/lapack_svd.h"
#include "simulator/mps/detail/linalg/matrix.h"
#include "simulator/mps/detail/linalg/threading.h"
#include "simulator/mps/compiler/mps_compiler.h"
#include "simulator/mps/mps_state.h"

namespace mindquantum::sim::mps::detail {
namespace detail {
template <typename T>
using Mat = linalg::Matrix<T>;

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
    linalg::SvdWorkspace<T> svd;

    void EnsureCapacity(size_t bond_dim_cap) {
        // Allocate for worst-case local sizes:
        //   L: (2*D_L) x D_M, R: D_M x (2*D_R), Theta: (2*D_L) x (2*D_R)
        // with D_L,D_M,D_R <= bond_dim_cap.
        const auto cap = static_cast<size_t>(bond_dim_cap);
        if (cap == 0) {
            return;
        }
        const size_t two_cap = 2 * cap;

        if (L.storage_size() < two_cap * cap) {
            L.reserve(two_cap, cap);
        }
        if (R.storage_size() < cap * two_cap) {
            R.reserve(cap, two_cap);
        }
        if (Theta.storage_size() < two_cap * two_cap) {
            Theta.reserve(two_cap, two_cap);
        }
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
auto MPSState<qs_policy_t_>::GammaView0(const GammaSite& site) -> linalg::MatrixView<const complex_t> {
    if (site.dl == 0 || site.dr == 0) {
        throw std::runtime_error("mqmps: invalid GammaSite dimensions (zero bond dimension).");
    }
    if (site.dl != 0 && site.dr > std::numeric_limits<size_t>::max() / site.dl) {
        throw std::runtime_error("mqmps: GammaSite size overflow.");
    }
    const size_t expected = site.dl * site.dr;
    if (site.g0.size() != expected) {
        throw std::runtime_error("mqmps: GammaSite g0 buffer size mismatch.");
    }
    return linalg::MatrixView<const complex_t>(site.g0.data(), site.dl, site.dr, site.dl);
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::GammaView1(const GammaSite& site) -> linalg::MatrixView<const complex_t> {
    if (site.dl == 0 || site.dr == 0) {
        throw std::runtime_error("mqmps: invalid GammaSite dimensions (zero bond dimension).");
    }
    if (site.dl != 0 && site.dr > std::numeric_limits<size_t>::max() / site.dl) {
        throw std::runtime_error("mqmps: GammaSite size overflow.");
    }
    const size_t expected = site.dl * site.dr;
    if (site.g1.size() != expected) {
        throw std::runtime_error("mqmps: GammaSite g1 buffer size mismatch.");
    }
    return linalg::MatrixView<const complex_t>(site.g1.data(), site.dl, site.dr, site.dl);
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::GammaView0Mut(GammaSite& site) -> linalg::MatrixView<complex_t> {
    if (site.dl == 0 || site.dr == 0) {
        throw std::runtime_error("mqmps: invalid GammaSite dimensions (zero bond dimension).");
    }
    if (site.dl != 0 && site.dr > std::numeric_limits<size_t>::max() / site.dl) {
        throw std::runtime_error("mqmps: GammaSite size overflow.");
    }
    const size_t expected = site.dl * site.dr;
    if (site.g0.size() != expected) {
        throw std::runtime_error("mqmps: GammaSite g0 buffer size mismatch.");
    }
    return linalg::MatrixView<complex_t>(site.g0.data(), site.dl, site.dr, site.dl);
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::GammaView1Mut(GammaSite& site) -> linalg::MatrixView<complex_t> {
    if (site.dl == 0 || site.dr == 0) {
        throw std::runtime_error("mqmps: invalid GammaSite dimensions (zero bond dimension).");
    }
    if (site.dl != 0 && site.dr > std::numeric_limits<size_t>::max() / site.dl) {
        throw std::runtime_error("mqmps: GammaSite size overflow.");
    }
    const size_t expected = site.dl * site.dr;
    if (site.g1.size() != expected) {
        throw std::runtime_error("mqmps: GammaSite g1 buffer size mismatch.");
    }
    return linalg::MatrixView<complex_t>(site.g1.data(), site.dl, site.dr, site.dl);
}

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
    (void) qubits_limit;
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

        const auto g0 = GammaView0(site);
        const auto g1 = GammaView1(site);

        Mat a0(acc.rows(), dr);
        Mat a1(acc.rows(), dr);
        linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(acc).view(), g0, complex_t{0, 0}, a0.view());
        linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(acc).view(), g1, complex_t{0, 0}, a1.view());

        // Scale columns by lambda[p+1]
        for (size_t j = 0; j < dr; ++j) {
            const auto s = lam_next[j];
            for (size_t i = 0; i < a0.rows(); ++i) {
                a0(i, j) *= s;
                a1(i, j) *= s;
            }
        }

        Mat next(2 * acc.rows(), dr);
        for (size_t j = 0; j < dr; ++j) {
            for (size_t i = 0; i < acc.rows(); ++i) {
                next(i, j) = a0(i, j);
                next(i + acc.rows(), j) = a1(i, j);
            }
        }
        acc.swap(next);
    }

    if (acc.cols() != 1) {
        throw std::runtime_error("mqmps: final bond dimension is not 1; cannot convert to full state vector.");
    }
    VT<complex_t> out(static_cast<size_t>(acc.rows()));
    for (size_t i = 0; i < acc.rows(); ++i) {
        out.at(i) = acc(i, 0);
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
            "mqmps: GetQS() constructs a full 2^n statevector and is limited by max output bytes; n="
            + std::to_string(n_qubits_) + ", elements=" + std::to_string(dim)
            + ", bytes_per_element=" + std::to_string(sizeof(complex_t))
            + ", limit=" + std::to_string(detail::kMaxDenseStateVectorOutputBytes) + " bytes.");
    }
    auto phys = ContractToStateVectorPhysicalOrder_();
    auto logical = PermutePhysicalToLogical_(phys);
    return logical;
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::SetFromStateVectorPhysical_(const VT<complex_t>& vec_phys) {
    using Mat = detail::Mat<complex_t>;
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
    Mat B(1, dim);
    for (size_t i = 0; i < dim; ++i) {
        B(0, i) = vec_phys.at(i);
    }

    for (qbit_t i = 0; i < n_qubits_; ++i) {
        const size_t D_i = lambdas_.at(static_cast<size_t>(i)).size();
        const size_t remaining = static_cast<size_t>(n_qubits_ - i);
        const size_t cols = static_cast<size_t>(1ULL) << remaining;
        const size_t cols_rest = cols / 2;

        if (static_cast<size_t>(B.rows()) != D_i || static_cast<size_t>(B.cols()) != cols) {
            throw std::runtime_error("mqmps: internal SetQS reshape mismatch.");
        }

        Mat M(2 * D_i, cols_rest);
        for (size_t cr = 0; cr < cols_rest; ++cr) {
            const size_t c0 = 2 * cr;
            const size_t c1 = 2 * cr + 1;
            for (size_t a = 0; a < D_i; ++a) {
                M(a, cr) = B(a, c0);
                M(a + D_i, cr) = B(a, c1);
            }
        }

        thread_local linalg::SvdWorkspace<complex_t> svd_ws;
        linalg::ComputeThinSvd(&M, &svd_ws);

        const auto& s_all = svd_ws.S;
        const auto r_full = s_all.size();
        size_t r_keep = r_full;
        if (max_bond_dimension_ != 0) {
            r_keep = std::min(r_keep, static_cast<size_t>(max_bond_dimension_));
        }
        if (r_keep == 0) {
            r_keep = 1;
        }
        if (r_keep > 1 && s_all.at(0) > 0) {
            for (size_t k = 0; k < r_keep; ++k) {
                if (s_all.at(k) / s_all.at(0) < eps_rel) {
                    r_keep = std::max<size_t>(1, k);
                    break;
                }
            }
        }

        // Copy + normalize kept singular values.
        VT<calc_type> s_kept(r_keep);
        calc_type s_norm2 = 0;
        for (size_t k = 0; k < r_keep; ++k) {
            s_kept[k] = s_all.at(k);
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

        const auto& U = svd_ws.U;
        const auto& lam_left = lambdas_.at(static_cast<size_t>(i));
        VT<calc_type> inv_left(D_i);
        for (size_t a = 0; a < D_i; ++a) {
            inv_left[a] = detail::SafeInv(lam_left.at(a), lambda_floor);
        }

        auto g0 = GammaView0Mut(site);
        auto g1 = GammaView1Mut(site);
        for (size_t k = 0; k < D_next; ++k) {
            for (size_t a = 0; a < D_i; ++a) {
                const auto inv = inv_left[a];
                g0(a, k) = U(a, k) * inv;
                g1(a, k) = U(a + D_i, k) * inv;
            }
        }

        // Build next B = diag(S) * Vh
        const auto& Vh = svd_ws.Vh;  // (r_full x cols_rest)
        Mat B_next(D_next, cols_rest);
        for (size_t c = 0; c < cols_rest; ++c) {
            for (size_t k = 0; k < D_next; ++k) {
                const calc_type s_k = lambdas_.at(static_cast<size_t>(i + 1)).at(k);
                B_next(k, c) = static_cast<complex_t>(s_k) * Vh(k, c);
            }
        }
        B.swap(B_next);
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
        throw std::runtime_error("mqmps: SetQS() requires a full 2^n statevector and is limited by max output bytes; n="
                                 + std::to_string(n_qubits_) + ", elements=" + std::to_string(dim)
                                 + ", bytes_per_element=" + std::to_string(sizeof(complex_t))
                                 + ", limit=" + std::to_string(detail::kMaxDenseStateVectorOutputBytes) + " bytes.");
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
void MPSState<qs_policy_t_>::Apply2QOnBond_(qbit_t bond, const std::array<complex_t, 16>& u, bool renormalize) {
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
    L.resize(2 * D_L, D_M);
    R.resize(D_M, 2 * D_R);

    const auto& lam_left = lambdas_.at(static_cast<size_t>(i));
    const auto& lam_mid = lambdas_.at(static_cast<size_t>(i + 1));
    const auto& lam_right = lambdas_.at(static_cast<size_t>(i + 2));
    const auto gi0 = GammaView0(site_i);
    const auto gi1 = GammaView1(site_i);
    const auto gj0 = GammaView0(site_j);
    const auto gj1 = GammaView1(site_j);

    // Build L (column-major: fill each column contiguously).
    for (size_t m = 0; m < D_M; ++m) {
        const calc_type lam_m = lam_mid.at(m);
        for (size_t a = 0; a < D_L; ++a) {
            const auto scale = lam_left.at(a) * lam_m;
            L(a, m) = gi0(a, m) * scale;
        }
        for (size_t a = 0; a < D_L; ++a) {
            const auto scale = lam_left.at(a) * lam_m;
            L(a + D_L, m) = gi1(a, m) * scale;
        }
    }

    // Build R (column-major: fill each column contiguously).
    for (size_t r = 0; r < D_R; ++r) {
        const calc_type lam_r = lam_right.at(r);
        for (size_t m = 0; m < D_M; ++m) {
            R(m, r) = gj0(m, r) * lam_r;
        }
        for (size_t m = 0; m < D_M; ++m) {
            R(m, r + D_R) = gj1(m, r) * lam_r;
        }
    }

    Theta.resize(2 * D_L, 2 * D_R);
    linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(L).view(), static_cast<const Mat&>(R).view(),
                 complex_t{0, 0}, Theta.view());

    // Apply 4x4 gate to physical indices (unrolled).
    for (size_t r = 0; r < D_R; ++r) {
        for (size_t a = 0; a < D_L; ++a) {
            const auto t00 = Theta(a, r);
            const auto t01 = Theta(a, r + D_R);
            const auto t10 = Theta(a + D_L, r);
            const auto t11 = Theta(a + D_L, r + D_R);

            const auto v00 = u[0] * t00 + u[1] * t01 + u[2] * t10 + u[3] * t11;
            const auto v01 = u[4] * t00 + u[5] * t01 + u[6] * t10 + u[7] * t11;
            const auto v10 = u[8] * t00 + u[9] * t01 + u[10] * t10 + u[11] * t11;
            const auto v11 = u[12] * t00 + u[13] * t01 + u[14] * t10 + u[15] * t11;

            Theta(a, r) = v00;
            Theta(a, r + D_R) = v01;
            Theta(a + D_L, r) = v10;
            Theta(a + D_L, r + D_R) = v11;
        }
    }

    linalg::ComputeThinSvd(&Theta, &ws.svd);
    const auto& s_all = ws.svd.S;
    const size_t r_full = s_all.size();
    size_t r_keep = r_full;
    if (max_bond_dimension_ != 0) {
        r_keep = std::min(r_keep, static_cast<size_t>(max_bond_dimension_));
    }
    if (r_keep == 0) {
        r_keep = 1;
    }
    if (r_keep > 1 && s_all[0] > 0) {
        for (size_t k = 0; k < r_keep; ++k) {
            if (s_all[k] / s_all[0] < eps_rel) {
                r_keep = std::max<size_t>(1, k);
                break;
            }
        }
    }

    VT<calc_type> s_kept(r_keep);
    for (size_t k = 0; k < r_keep; ++k) {
        s_kept[k] = s_all[k];
    }
    if (renormalize) {
        // Keep the MPS state normalized when simulating unitary circuits (and when truncation is enabled).
        //
        // For non-unitary operations (e.g., `diff=True` derivative gates), the norm carries meaning and must be
        // preserved.
        calc_type norm2 = 0;
        for (size_t k = 0; k < r_keep; ++k) {
            norm2 += s_kept[k] * s_kept[k];
        }
        const calc_type norm = std::sqrt(norm2);
        if (norm > 0) {
            for (auto& v : s_kept) {
                v /= norm;
            }
        }
    }

    lambdas_.at(static_cast<size_t>(i + 1)) = std::move(s_kept);

    const size_t D_new = lambdas_.at(static_cast<size_t>(i + 1)).size();
    const auto& U = ws.svd.U;
    const auto& Vh = ws.svd.Vh;  // (r_full x 2*D_R)

    // Update left site gamma (D_L x D_new)
    site_i.dr = D_new;
    site_i.g0.resize(D_L * D_new);
    site_i.g1.resize(D_L * D_new);
    VT<calc_type> inv_left(D_L);
    for (size_t a = 0; a < D_L; ++a) {
        inv_left[a] = detail::SafeInv(lam_left.at(a), lambda_floor);
    }
    auto site_i_g0 = GammaView0Mut(site_i);
    auto site_i_g1 = GammaView1Mut(site_i);
    for (size_t k = 0; k < D_new; ++k) {
        for (size_t a = 0; a < D_L; ++a) {
            const auto inv = inv_left[a];
            site_i_g0(a, k) = U(a, k) * inv;
            site_i_g1(a, k) = U(a + D_L, k) * inv;
        }
    }

    // Update right site gamma (D_new x D_R) from Vh blocks, divide by lambda[i+2]
    site_j.dl = D_new;
    site_j.g0.resize(D_new * D_R);
    site_j.g1.resize(D_new * D_R);
    auto site_j_g0 = GammaView0Mut(site_j);
    auto site_j_g1 = GammaView1Mut(site_j);
    for (size_t r = 0; r < D_R; ++r) {
        const calc_type inv = detail::SafeInv(lam_right.at(r), lambda_floor);
        for (size_t k = 0; k < D_new; ++k) {
            site_j_g0(k, r) = Vh(k, r) * inv;
            site_j_g1(k, r) = Vh(k, r + D_R) * inv;
        }
    }
}

template <typename qs_policy_t_>
void MPSState<qs_policy_t_>::ReorderToTargetMapping_(const qbits_t& logical_at_pos_target) {
    if (logical_at_pos_target.size() != static_cast<size_t>(n_qubits_)) {
        throw std::runtime_error("mqmps: invalid target mapping size.");
    }
    if (logical_at_pos_.size() != static_cast<size_t>(n_qubits_)
        || pos_of_logical_.size() != static_cast<size_t>(n_qubits_)) {
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
    linalg::ScopedOpenBlasNumThreads openblas_threads(1);
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
                Apply2QOnBond_(op.bond, op.mat, !diff);
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

            // Sample full register bits in physical site order.
            VT<int> bits(static_cast<size_t>(n_qubits_), 0);
            if (n_qubits_ > 0) {
                Mat x(1, 1);
                x(0, 0) = complex_t{1, 0};
                for (qbit_t i = 0; i < n_qubits_; ++i) {
                    const auto D_i = lambdas_.at(static_cast<size_t>(i)).size();
                    const auto D_next = lambdas_.at(static_cast<size_t>(i + 1)).size();
                    const auto& lam_next = lambdas_.at(static_cast<size_t>(i + 1));
                    const auto& site = gammas_.at(static_cast<size_t>(i));
                    if (site.dl != D_i || site.dr != D_next) {
                        throw std::runtime_error("mqmps: inconsistent dimensions in measurement sampling.");
                    }
                    const auto g0 = GammaView0(site);
                    const auto g1 = GammaView1(site);
                    if (x.cols() != D_i) {
                        throw std::runtime_error("mqmps: internal measurement sampling dim mismatch.");
                    }
                    Mat y0(1, D_next);
                    Mat y1(1, D_next);
                    linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(x).view(), g0, complex_t{0, 0},
                                 y0.view());
                    linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(x).view(), g1, complex_t{0, 0},
                                 y1.view());

                    for (size_t k = 0; k < D_next; ++k) {
                        y0(0, k) *= lam_next[k];
                        y1(0, k) *= lam_next[k];
                    }

                    double p0 = 0;
                    double p1 = 0;
                    for (size_t k = 0; k < D_next; ++k) {
                        p0 += static_cast<double>(std::norm(y0(0, k)));
                        p1 += static_cast<double>(std::norm(y1(0, k)));
                    }
                    const double denom = p0 + p1;
                    if (denom <= 0) {
                        throw std::runtime_error("mqmps: invalid probability in measurement sampling.");
                    }
                    const double prob0 = p0 / denom;
                    const bool bit0 = (rng_() < prob0);
                    bits.at(static_cast<size_t>(i)) = bit0 ? 0 : 1;
                    if (bit0) {
                        x.swap(y0);
                    } else {
                        x.swap(y1);
                    }
                }
            }

            // Apply measurement gates sequentially to support multiple measurements on the same qubit
            // (e.g. measure with reset_to, then measure again to verify the reset).
            //
            // A repeated measurement on a collapsed/reset qubit is deterministic; we track the current
            // post-operation bit value for each physical site to assign outcomes to subsequent measure keys.
            VT<uint8_t> has_known_bit(static_cast<size_t>(n_qubits_), 0);
            VT<int> current_bit(static_cast<size_t>(n_qubits_), 0);

            for (const auto& m : stage.meas) {
                if (m.pos < 0 || m.pos >= n_qubits_) {
                    throw std::runtime_error("mqmps: invalid measurement site index.");
                }
                const auto pos = static_cast<size_t>(m.pos);

                int outcome = 0;
                if (has_known_bit.at(pos) == 0) {
                    outcome = bits.at(pos);
                    has_known_bit.at(pos) = 1;
                    current_bit.at(pos) = outcome;

                    // Collapse (unnormalized) on first measurement for this site.
                    auto& site = gammas_.at(pos);
                    if (outcome == 0) {
                        std::fill(site.g1.begin(), site.g1.end(), complex_t{0, 0});
                    } else {
                        std::fill(site.g0.begin(), site.g0.end(), complex_t{0, 0});
                    }
                } else {
                    // Deterministic: state on this site is already |current_bit⟩ after collapse/reset.
                    outcome = current_bit.at(pos);
                }

                meas_result[m.key] = outcome;

                // Apply post-measure reset as a physical X (swap Γ0/Γ1).
                if (m.reset && static_cast<index_t>(outcome) != m.reset_to) {
                    auto& site = gammas_.at(pos);
                    std::swap(site.g0, site.g1);
                    current_bit.at(pos) = static_cast<int>(m.reset_to);
                }
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
                const std::array<complex_t, 16> eye4 = {
                    complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{0, 0},
                    complex_t{0, 0}, complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0},
                    complex_t{0, 0}, complex_t{0, 0}, complex_t{1, 0}, complex_t{0, 0},
                    complex_t{0, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{1, 0}};
                for (qbit_t bond = 0; bond < n_qubits_ - 1; ++bond) {
                    Apply2QOnBond_(bond, eye4);
                }
            }
        }
    }

    return meas_result;
}

template <typename qs_policy_t_>
index_t MPSState<qs_policy_t_>::ApplyGate(const std::shared_ptr<BasicGate>& gate,
                                          const parameter::ParameterResolver& pr, bool diff) {
    if (gate->GetID() == GateID::M) {
        auto res = ApplyCircuitCompiled_({gate}, pr, false);
        if (res.empty()) {
            return 2;
        }
        return static_cast<index_t>(res.begin()->second);
    }
    ApplyCircuitCompiled_({gate}, pr, diff);
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
auto MPSState<qs_policy_t_>::SamplingMeasurementEndingWithoutNoise(const circuit_t& circ,
                                                                   const parameter::ParameterResolver& pr, size_t shots,
                                                                   const MST<size_t>& key_map, unsigned seed) const
    -> VT<unsigned> {
    using Mat = detail::Mat<complex_t>;

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
    Mat tmp;
    for (qbit_t i = sim.n_qubits_ - 1; i >= 0; --i) {
        const auto D_i = sim.lambdas_.at(static_cast<size_t>(i)).size();
        const auto D_next = sim.lambdas_.at(static_cast<size_t>(i + 1)).size();
        const auto& lam_next = sim.lambdas_.at(static_cast<size_t>(i + 1));
        const auto& site = sim.gammas_.at(static_cast<size_t>(i));
        if (site.dl != D_i || site.dr != D_next) {
            throw std::runtime_error("mqmps: inconsistent dimensions in sampling env.");
        }
        const auto g0 = GammaView0(site);
        const auto g1 = GammaView1(site);
        const auto& Rn = R_env.at(static_cast<size_t>(i + 1));
        Mat Rscaled = Rn;
        // Rscaled = diag(lambda_next) * Rn * diag(lambda_next)
        linalg::ScaleByDiagonals(Rscaled.view(), lam_next, lam_next, "sampling env Rscaled");

        Mat Ri(D_i, D_i);
        tmp.resize(D_i, D_next);

        // Ri = g0 * Rscaled * g0^H
        linalg::Gemm('N', 'N', complex_t{1, 0}, g0, static_cast<const Mat&>(Rscaled).view(), complex_t{0, 0},
                     tmp.view());
        linalg::Gemm('N', 'C', complex_t{1, 0}, static_cast<const Mat&>(tmp).view(), g0, complex_t{0, 0}, Ri.view());

        // Ri += g1 * Rscaled * g1^H
        linalg::Gemm('N', 'N', complex_t{1, 0}, g1, static_cast<const Mat&>(Rscaled).view(), complex_t{0, 0},
                     tmp.view());
        linalg::Gemm('N', 'C', complex_t{1, 0}, static_cast<const Mat&>(tmp).view(), g1, complex_t{1, 0}, Ri.view());

        R_env.at(static_cast<size_t>(i)) = std::move(Ri);
    }

    const auto key_size = key_map.size();
    VT<unsigned> out(shots * key_size);

    for (size_t shot = 0; shot < shots; ++shot) {
        // Sample full register bits in physical site order.
        VT<int> bits(static_cast<size_t>(sim.n_qubits_), 0);
        Mat x(1, 1);
        x(0, 0) = complex_t{1, 0};
        Mat y0;
        Mat y1;
        Mat tmp_vec;
        Mat scalar(1, 1);
        for (qbit_t i = 0; i < sim.n_qubits_; ++i) {
            const auto D_i = sim.lambdas_.at(static_cast<size_t>(i)).size();
            const auto D_next = sim.lambdas_.at(static_cast<size_t>(i + 1)).size();
            const auto& lam_next = sim.lambdas_.at(static_cast<size_t>(i + 1));
            const auto& site = sim.gammas_.at(static_cast<size_t>(i));
            if (site.dl != D_i || site.dr != D_next) {
                throw std::runtime_error("mqmps: inconsistent dimensions in sampling.");
            }
            const auto g0 = GammaView0(site);
            const auto g1 = GammaView1(site);
            if (x.cols() != D_i) {
                throw std::runtime_error("mqmps: internal sampling left-vector dim mismatch.");
            }
            y0.resize(1, D_next);
            y1.resize(1, D_next);
            linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(x).view(), g0, complex_t{0, 0},
                         y0.view());
            linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(x).view(), g1, complex_t{0, 0},
                         y1.view());
            for (size_t k = 0; k < D_next; ++k) {
                y0(0, k) *= lam_next[k];
                y1(0, k) *= lam_next[k];
            }
            const auto& Rn = R_env.at(static_cast<size_t>(i + 1));

            tmp_vec.resize(D_next, 1);
            linalg::Gemm('N', 'C', complex_t{1, 0}, static_cast<const Mat&>(Rn).view(), static_cast<const Mat&>(y0).view(),
                         complex_t{0, 0}, tmp_vec.view());
            linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(y0).view(),
                         static_cast<const Mat&>(tmp_vec).view(), complex_t{0, 0}, scalar.view());
            const complex_t p0 = scalar(0, 0);

            linalg::Gemm('N', 'C', complex_t{1, 0}, static_cast<const Mat&>(Rn).view(), static_cast<const Mat&>(y1).view(),
                         complex_t{0, 0}, tmp_vec.view());
            linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(y1).view(),
                         static_cast<const Mat&>(tmp_vec).view(), complex_t{0, 0}, scalar.view());
            const complex_t p1 = scalar(0, 0);

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
            if (bit0) {
                x.swap(y0);
            } else {
                x.swap(y1);
            }
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
            "mqmps: reduced density matrix is limited by max output bytes; kept_qubits=" + std::to_string(k)
            + ", elements=" + std::to_string(dim_rho) + ", bytes_per_element=" + std::to_string(sizeof(complex_t))
            + ", limit=" + std::to_string(detail::kMaxReducedDensityMatrixOutputBytes) + " bytes.");
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
    linalg::ScopedOpenBlasNumThreads openblas_threads(1);

    const auto scale_by_lambda = [](const VT<calc_type>& lam_next, Mat* m) {
        linalg::ScaleByDiagonals(m->view(), lam_next, lam_next, "reduced density matrix lambda scaling");
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

        const auto g0 = GammaView0(g);
        const auto g1 = GammaView1(g);

        a0->resize(D_i, D_next);
        a1->resize(D_i, D_next);
        linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(*e).view(), g0, complex_t{0, 0}, a0->view());
        linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(*e).view(), g1, complex_t{0, 0}, a1->view());

        tmp->resize(D_next, D_next);
        linalg::Gemm('C', 'N', complex_t{1, 0}, g0, static_cast<const Mat&>(*a0).view(), complex_t{0, 0},
                     tmp->view());
        linalg::Gemm('C', 'N', complex_t{1, 0}, g1, static_cast<const Mat&>(*a1).view(), complex_t{1, 0},
                     tmp->view());
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

        const auto g0 = GammaView0(g);
        const auto g1 = GammaView1(g);

        Mat& ek0 = ws->ket0_stack[depth];
        Mat& ek1 = ws->ket1_stack[depth];
        ek0.resize(D_i, D_next);
        ek1.resize(D_i, D_next);
        linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(e).view(), g0, complex_t{0, 0}, ek0.view());
        linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(e).view(), g1, complex_t{0, 0}, ek1.view());

        for (size_t ket_bit = 0; ket_bit < 2; ++ket_bit) {
            const Mat& tmp_k = (ket_bit == 0) ? ek0 : ek1;
            const size_t next_row = row_acc | (ket_bit << log_bit);
            for (size_t bra_bit = 0; bra_bit < 2; ++bra_bit) {
                const auto& gl = (bra_bit == 0) ? g0 : g1;
                const size_t next_col = col_acc | (bra_bit << log_bit);

                Mat& child = ws->env_stack[depth + 1];
                child.resize(D_next, D_next);
                linalg::Gemm('C', 'N', complex_t{1, 0}, gl, static_cast<const Mat&>(tmp_k).view(), complex_t{0, 0},
                             child.view());
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

        const auto g0 = GammaView0(g);
        const auto g1 = GammaView1(g);

        // Reuse a0/a1 as env*g0/env*g1 at the kept site.
        a0.resize(D_i, D_next);
        a1.resize(D_i, D_next);
        linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(env).view(), g0, complex_t{0, 0}, a0.view());
        linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(env).view(), g1, complex_t{0, 0}, a1.view());

        const size_t next_depth = depth + 1;
        const qbit_t next_site_child = kept_pos + 1;
        for (size_t ket_bit = 0; ket_bit < 2; ++ket_bit) {
            const Mat& tmp_k = (ket_bit == 0) ? a0 : a1;
            const size_t next_row = row_acc | (ket_bit << log_bit);
            for (size_t bra_bit = 0; bra_bit < 2; ++bra_bit) {
                const auto& gl = (bra_bit == 0) ? g0 : g1;
                const size_t next_col = col_acc | (bra_bit << log_bit);

                Mat child(D_next, D_next);
                linalg::Gemm('C', 'N', complex_t{1, 0}, gl, static_cast<const Mat&>(tmp_k).view(), complex_t{0, 0},
                             child.view());
                scale_by_lambda(lam_next, &child);

#pragma omp task firstprivate(child, next_depth, next_site_child, next_row, next_col)
                {
                    dfs_parallel(next_depth, next_site_child, std::move(child), next_row, next_col);
                }
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
            {
                dfs_parallel(0, 0, std::move(init_env), 0, 0);
            }
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

        const auto g0 = GammaView0(g);
        const auto g1 = GammaView1(g);

        ws.a0.resize(D_i, D_next);
        ws.a1.resize(D_i, D_next);
        linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(e).view(), g0, complex_t{0, 0}, ws.a0.view());
        linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(e).view(), g1, complex_t{0, 0}, ws.a1.view());

        out->resize(D_next, D_next);

        // out = o00*(g0^H a0) + o01*(g0^H a1) + o10*(g1^H a0) + o11*(g1^H a1)
        linalg::Gemm('C', 'N', op[0], g0, static_cast<const Mat&>(ws.a0).view(), complex_t{0, 0}, out->view());
        if (op[1] != complex_t{0, 0}) {
            linalg::Gemm('C', 'N', op[1], g0, static_cast<const Mat&>(ws.a1).view(), complex_t{1, 0}, out->view());
        }
        if (op[2] != complex_t{0, 0}) {
            linalg::Gemm('C', 'N', op[2], g1, static_cast<const Mat&>(ws.a0).view(), complex_t{1, 0}, out->view());
        }
        if (op[3] != complex_t{0, 0}) {
            linalg::Gemm('C', 'N', op[3], g1, static_cast<const Mat&>(ws.a1).view(), complex_t{1, 0}, out->view());
        }

        // out = diag(lam_next) * out * diag(lam_next)
        linalg::ScaleByDiagonals(out->view(), lam_next, lam_next, "expectation lambda scaling");
    };

    return detail::EvaluateMergedPauliSumWithPrefixReuse<complex_t, calc_type>(
        n_qubits_, pauli_sum_cache_->phys_terms, eye, apply_local, get_pauli_op, []() { return complex_t{1, 0}; },
        false);
}

template <typename qs_policy_t_>
auto MPSState<qs_policy_t_>::ExpectationPauliTermsBraKet_(const VT<PauliTerm<calc_type>>& terms,
                                                          const derived_t& bra) const -> py_qs_data_t {
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

        const auto g0b = GammaView0(gb);
        const auto g1b = GammaView1(gb);
        const auto g0k = GammaView0(gk);
        const auto g1k = GammaView1(gk);

        ws.ek0.resize(D_b, D_k_next);
        ws.ek1.resize(D_b, D_k_next);
        linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(e_in).view(), g0k, complex_t{0, 0},
                     ws.ek0.view());
        linalg::Gemm('N', 'N', complex_t{1, 0}, static_cast<const Mat&>(e_in).view(), g1k, complex_t{0, 0},
                     ws.ek1.view());

        out->resize(D_b_next, D_k_next);
        out->set_zero();

        if (op[0] != complex_t{0, 0}) {
            linalg::Gemm('C', 'N', op[0], g0b, static_cast<const Mat&>(ws.ek0).view(), complex_t{1, 0}, out->view());
        }
        if (op[1] != complex_t{0, 0}) {
            linalg::Gemm('C', 'N', op[1], g0b, static_cast<const Mat&>(ws.ek1).view(), complex_t{1, 0}, out->view());
        }
        if (op[2] != complex_t{0, 0}) {
            linalg::Gemm('C', 'N', op[2], g1b, static_cast<const Mat&>(ws.ek0).view(), complex_t{1, 0}, out->view());
        }
        if (op[3] != complex_t{0, 0}) {
            linalg::Gemm('C', 'N', op[3], g1b, static_cast<const Mat&>(ws.ek1).view(), complex_t{1, 0}, out->view());
        }

        // out = diag(lam_b_next) * out * diag(lam_k_next)
        linalg::ScaleByDiagonals(out->view(), lam_b_next, lam_k_next, "bra/ket expectation lambda scaling");
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

    return detail::EvaluateMergedPauliSumWithPrefixReuse<complex_t, calc_type>(n_qubits_, merged, eye, apply_local,
                                                                               get_pauli_op, inner_product, true);
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
                                            const circuit_t& circ_left, const parameter::ParameterResolver& pr) const
    -> py_qs_data_t {
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
    (void) ham;
    throw std::runtime_error("mqmps: apply_hamiltonian is not supported (native MPO/apply path not implemented).");
}

template <typename qs_policy_t_>
VVT<typename MPSState<qs_policy_t_>::py_qs_data_t> MPSState<qs_policy_t_>::GetCircuitMatrix(
    const circuit_t& circ, const parameter::ParameterResolver& pr) const {
    (void) circ;
    (void) pr;
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
