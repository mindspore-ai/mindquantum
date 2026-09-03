/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef INCLUDE_SIMULATOR_CIRCUIT_TILING_DETAIL_KERNELS_H_
#define INCLUDE_SIMULATOR_CIRCUIT_TILING_DETAIL_KERNELS_H_

#include <cmath>

#include <cassert>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <variant>

#include "simulator/circuit_tiling/circuit_tiling.h"

namespace mindquantum::sim::circuit_tiling::detail {
// backend_t is a zero-overhead numerical algebra.  Scalar and ISA-specific
// implementations provide only predicates, cached loads/stores, and complex
// arithmetic primitives; all quantum-operation semantics live here once.
template <typename backend_t>
class Kernels {
 public:
    using value_t = typename backend_t::value_t;
    using vector_t = typename backend_t::vector_t;
    using predicate_t = typename backend_t::predicate_t;
    using state_index_t = typename backend_t::state_index_t;
    using index_context_t = typename backend_t::index_context_t;
    using storage_t = typename backend_t::storage_t;
    using calc_type = typename value_t::value_type;

    static void Apply(const Operation<value_t>& operation, predicate_t active, state_index_t state_base,
                      const index_context_t& index_context, storage_t* tile, storage_t* workspace) {
        const auto predicate = backend_t::ExternalControls(active, state_base, index_context,
                                                           operation.external_control_mask);
        if (backend_t::Skip(predicate)) {
            return;
        }
        std::visit(
            [&](const auto& kernel) { ApplyKernel(kernel, predicate, state_base, index_context, tile, workspace); },
            operation.kernel);
    }

    [[nodiscard]] static auto ExpectDerivative(const Derivative<value_t>& derivative, predicate_t active,
                                               state_index_t state_base, const index_context_t& index_context,
                                               const storage_t* bra, const storage_t* ket) -> value_t {
        const auto predicate = backend_t::ExternalControls(active, state_base, index_context,
                                                           derivative.external_control_mask);
        if (backend_t::Skip(predicate)) {
            return 0;
        }
        return std::visit(
            [&](const auto& kernel) {
                return backend_t::Reduce(predicate,
                                         ExpectKernel(kernel, predicate, state_base, index_context, bra, ket));
            },
            derivative.kernel);
    }

    [[nodiscard]] static auto InnerProduct(predicate_t predicate, const storage_t* bra, const storage_t* ket,
                                           size_t dimension) -> value_t {
        auto result = backend_t::Zero();
        for (size_t basis = 0; basis < dimension; ++basis) {
            result = backend_t::InnerProductAdd(predicate, result, backend_t::Load(bra, basis),
                                                backend_t::Load(ket, basis));
        }
        return backend_t::Reduce(predicate, result);
    }

 private:
    static void ApplyKernel(const PauliKernel<value_t>& kernel, predicate_t predicate, state_index_t state_base,
                            const index_context_t& index_context, storage_t* tile, storage_t*) {
        if (kernel.external_z_mask == 0) {
            for (const auto& scale : kernel.scales) {
                backend_t::Store(
                    tile, scale.basis, predicate,
                    backend_t::MultiplyCoefficient(predicate, backend_t::Load(tile, scale.basis), scale.factor));
            }
            for (const auto& pair : kernel.pairs) {
                const auto lhs = backend_t::Load(tile, pair.lhs);
                const auto rhs = backend_t::Load(tile, pair.rhs);
                backend_t::Store(tile, pair.lhs, predicate,
                                 backend_t::MultiplyCoefficient(predicate, rhs, pair.lhs_from_rhs));
                backend_t::Store(tile, pair.rhs, predicate,
                                 backend_t::MultiplyCoefficient(predicate, lhs, pair.rhs_from_lhs));
            }
            return;
        }

        const auto negative = backend_t::ExternalParity(predicate, state_base, index_context, kernel.external_z_mask);
        for (const auto& scale : kernel.scales) {
            backend_t::Store(tile, scale.basis, predicate,
                             backend_t::MultiplyDynamic(predicate, backend_t::Load(tile, scale.basis),
                                                        backend_t::SignedCoefficient(scale.factor, negative)));
        }
        for (const auto& pair : kernel.pairs) {
            const auto lhs = backend_t::Load(tile, pair.lhs);
            const auto rhs = backend_t::Load(tile, pair.rhs);
            backend_t::Store(
                tile, pair.lhs, predicate,
                backend_t::MultiplyDynamic(predicate, rhs, backend_t::SignedCoefficient(pair.lhs_from_rhs, negative)));
            backend_t::Store(
                tile, pair.rhs, predicate,
                backend_t::MultiplyDynamic(predicate, lhs, backend_t::SignedCoefficient(pair.rhs_from_lhs, negative)));
        }
    }

    static void ApplyKernel(const PauliRotationKernel<value_t>& kernel, predicate_t predicate, state_index_t state_base,
                            const index_context_t& index_context, storage_t* tile, storage_t*) {
        if (kernel.coefficients_prepared) {
            for (const auto& scale : kernel.scales) {
                backend_t::Store(
                    tile, scale.basis, predicate,
                    backend_t::MultiplyCoefficient(predicate, backend_t::Load(tile, scale.basis), scale.factor));
            }
            for (const auto& pair : kernel.pairs) {
                const auto lhs = backend_t::Load(tile, pair.lhs);
                const auto rhs = backend_t::Load(tile, pair.rhs);
                auto out_lhs = backend_t::Multiply(predicate, lhs, kernel.cosine);
                auto out_rhs = backend_t::Multiply(predicate, rhs, kernel.cosine);
                out_lhs = backend_t::MultiplyAddCoefficient(predicate, out_lhs, rhs, pair.lhs_from_rhs);
                out_rhs = backend_t::MultiplyAddCoefficient(predicate, out_rhs, lhs, pair.rhs_from_lhs);
                backend_t::Store(tile, pair.lhs, predicate, out_lhs);
                backend_t::Store(tile, pair.rhs, predicate, out_rhs);
            }
            return;
        }

        const auto negative = backend_t::ExternalParity(predicate, state_base, index_context, kernel.external_z_mask);
        const auto minus_i_sine = value_t{0, -kernel.sine};
        for (const auto& scale : kernel.scales) {
            auto coefficient = backend_t::Broadcast(value_t{kernel.cosine, 0});
            coefficient = backend_t::Add(predicate, coefficient,
                                         backend_t::SignedCoefficient(minus_i_sine * scale.factor, negative));
            backend_t::Store(tile, scale.basis, predicate,
                             backend_t::MultiplyDynamic(predicate, backend_t::Load(tile, scale.basis), coefficient));
        }
        for (const auto& pair : kernel.pairs) {
            const auto lhs = backend_t::Load(tile, pair.lhs);
            const auto rhs = backend_t::Load(tile, pair.rhs);
            auto out_lhs = backend_t::Multiply(predicate, lhs, kernel.cosine);
            auto out_rhs = backend_t::Multiply(predicate, rhs, kernel.cosine);
            out_lhs = backend_t::MultiplyAddDynamic(
                predicate, out_lhs, rhs, backend_t::SignedCoefficient(minus_i_sine * pair.lhs_from_rhs, negative));
            out_rhs = backend_t::MultiplyAddDynamic(
                predicate, out_rhs, lhs, backend_t::SignedCoefficient(minus_i_sine * pair.rhs_from_lhs, negative));
            backend_t::Store(tile, pair.lhs, predicate, out_lhs);
            backend_t::Store(tile, pair.rhs, predicate, out_rhs);
        }
    }

    static void ApplyKernel(const DiagonalKernel<value_t>& kernel, predicate_t predicate, state_index_t,
                            const index_context_t&, storage_t* tile, storage_t*) {
        for (const auto& scale : kernel.scales) {
            backend_t::Store(
                tile, scale.basis, predicate,
                backend_t::MultiplyCoefficient(predicate, backend_t::Load(tile, scale.basis), scale.factor));
        }
    }

    static void ApplyKernel(const HadamardKernel& kernel, predicate_t predicate, state_index_t, const index_context_t&,
                            storage_t* tile, storage_t*) {
        constexpr auto scale = static_cast<calc_type>(M_SQRT1_2);
        for (const auto& pair : kernel.pairs) {
            const auto lhs = backend_t::Load(tile, pair.lhs);
            const auto rhs = backend_t::Load(tile, pair.rhs);
            backend_t::Store(tile, pair.lhs, predicate,
                             backend_t::Multiply(predicate, backend_t::Add(predicate, lhs, rhs), scale));
            backend_t::Store(tile, pair.rhs, predicate,
                             backend_t::Multiply(predicate, backend_t::Subtract(predicate, lhs, rhs), scale));
        }
    }

    static void ApplyKernel(const PairMatrixKernel<value_t>& kernel, predicate_t predicate, state_index_t,
                            const index_context_t&, storage_t* tile, storage_t*) {
        for (const auto& pair : kernel.pairs) {
            const auto lhs = backend_t::Load(tile, pair.lhs);
            const auto rhs = backend_t::Load(tile, pair.rhs);
            auto out_lhs = backend_t::Zero();
            auto out_rhs = backend_t::Zero();
            out_lhs = backend_t::MultiplyAddCoefficient(predicate, out_lhs, lhs, kernel.matrix[0]);
            out_lhs = backend_t::MultiplyAddCoefficient(predicate, out_lhs, rhs, kernel.matrix[1]);
            out_rhs = backend_t::MultiplyAddCoefficient(predicate, out_rhs, lhs, kernel.matrix[2]);
            out_rhs = backend_t::MultiplyAddCoefficient(predicate, out_rhs, rhs, kernel.matrix[3]);
            backend_t::Store(tile, pair.lhs, predicate, out_lhs);
            backend_t::Store(tile, pair.rhs, predicate, out_rhs);
        }
    }

    static void ApplyKernel(const SwapKernel& kernel, predicate_t predicate, state_index_t, const index_context_t&,
                            storage_t* tile, storage_t*) {
        for (const auto& pair : kernel.pairs) {
            const auto lhs = backend_t::Load(tile, pair.lhs);
            const auto rhs = backend_t::Load(tile, pair.rhs);
            backend_t::Store(tile, pair.lhs, predicate, rhs);
            backend_t::Store(tile, pair.rhs, predicate, lhs);
        }
    }

    static void ApplyKernel(const ISwapKernel<value_t>& kernel, predicate_t predicate, state_index_t,
                            const index_context_t&, storage_t* tile, storage_t*) {
        for (const auto& pair : kernel.pairs) {
            const auto lhs = backend_t::Load(tile, pair.lhs);
            const auto rhs = backend_t::Load(tile, pair.rhs);
            backend_t::Store(tile, pair.lhs, predicate, backend_t::MultiplyCoefficient(predicate, rhs, kernel.phase));
            backend_t::Store(tile, pair.rhs, predicate, backend_t::MultiplyCoefficient(predicate, lhs, kernel.phase));
        }
    }

    static void ApplyKernel(const FSimKernel<value_t>& kernel, predicate_t predicate, state_index_t,
                            const index_context_t&, storage_t* tile, storage_t*) {
        for (const auto& indices : kernel.indices) {
            const auto lhs = backend_t::Load(tile, indices.lhs);
            const auto rhs = backend_t::Load(tile, indices.rhs);
            auto out_lhs = backend_t::Zero();
            auto out_rhs = backend_t::Zero();
            out_lhs = backend_t::MultiplyAddCoefficient(predicate, out_lhs, lhs, kernel.exchange[0]);
            out_lhs = backend_t::MultiplyAddCoefficient(predicate, out_lhs, rhs, kernel.exchange[1]);
            out_rhs = backend_t::MultiplyAddCoefficient(predicate, out_rhs, lhs, kernel.exchange[2]);
            out_rhs = backend_t::MultiplyAddCoefficient(predicate, out_rhs, rhs, kernel.exchange[3]);
            backend_t::Store(tile, indices.lhs, predicate, out_lhs);
            backend_t::Store(tile, indices.rhs, predicate, out_rhs);
            backend_t::Store(
                tile, indices.both, predicate,
                backend_t::MultiplyCoefficient(predicate, backend_t::Load(tile, indices.both), kernel.phase));
        }
    }

    static void ApplyKernel(const DenseMatrixKernel<value_t>& kernel, predicate_t predicate, state_index_t,
                            const index_context_t&, storage_t* tile, storage_t* workspace) {
        assert(kernel.matrix_dim != 0 && kernel.indices.size() % kernel.matrix_dim == 0);
        for (size_t group = 0; group < kernel.indices.size(); group += kernel.matrix_dim) {
            for (size_t column = 0; column < kernel.matrix_dim; ++column) {
                backend_t::Store(workspace, column, predicate, backend_t::Load(tile, kernel.indices[group + column]));
            }
            for (size_t row = 0; row < kernel.matrix_dim; ++row) {
                auto output = backend_t::Zero();
                const auto* matrix_row = kernel.matrix.data() + row * kernel.matrix_dim;
                for (size_t column = 0; column < kernel.matrix_dim; ++column) {
                    output = backend_t::MultiplyAddCoefficient(predicate, output, backend_t::Load(workspace, column),
                                                               matrix_row[column]);
                }
                backend_t::Store(tile, kernel.indices[group + row], predicate, output);
            }
        }
    }

    static void ApplyKernel(const DepolarizingKernel<value_t>& kernel, predicate_t predicate, state_index_t,
                            const index_context_t&, storage_t* tile, storage_t*) {
        const auto block = kernel.local_dim * kernel.local_dim;
        assert(kernel.local_dim != 0 && kernel.indices.size() % block == 0);
        // Entry r * local_dim + c of a block is diagonal exactly when it is a
        // multiple of local_dim + 1.
        const auto diagonal_stride = kernel.local_dim + 1;
        for (size_t group = 0; group < kernel.indices.size(); group += block) {
            auto trace = backend_t::Zero();
            for (size_t basis = 0; basis < kernel.local_dim; ++basis) {
                trace = backend_t::Add(predicate, trace,
                                       backend_t::Load(tile, kernel.indices[group + basis * diagonal_stride]));
            }
            const auto mixed = backend_t::Multiply(predicate, trace, kernel.mix);
            for (size_t entry = 0; entry < block; ++entry) {
                const auto index = kernel.indices[group + entry];
                auto value = backend_t::Multiply(predicate, backend_t::Load(tile, index), kernel.keep);
                if (entry % diagonal_stride == 0) {
                    value = backend_t::Add(predicate, value, mixed);
                }
                backend_t::Store(tile, index, predicate, value);
            }
        }
    }

    [[nodiscard]] static auto ExpectKernel(const PauliRotationKernel<value_t>& kernel, predicate_t predicate,
                                           state_index_t state_base, const index_context_t& index_context,
                                           const storage_t* bra, const storage_t* ket) -> vector_t {
        auto result = backend_t::Zero();
        if (kernel.coefficients_prepared) {
            for (const auto& scale : kernel.scales) {
                const auto transformed = backend_t::MultiplyCoefficient(predicate, backend_t::Load(ket, scale.basis),
                                                                        scale.factor);
                result = backend_t::InnerProductAdd(predicate, result, backend_t::Load(bra, scale.basis), transformed);
            }
            for (const auto& pair : kernel.pairs) {
                const auto ket_lhs = backend_t::Load(ket, pair.lhs);
                const auto ket_rhs = backend_t::Load(ket, pair.rhs);
                auto out_lhs = backend_t::Multiply(predicate, ket_lhs, kernel.cosine);
                auto out_rhs = backend_t::Multiply(predicate, ket_rhs, kernel.cosine);
                out_lhs = backend_t::MultiplyAddCoefficient(predicate, out_lhs, ket_rhs, pair.lhs_from_rhs);
                out_rhs = backend_t::MultiplyAddCoefficient(predicate, out_rhs, ket_lhs, pair.rhs_from_lhs);
                result = backend_t::InnerProductAdd(predicate, result, backend_t::Load(bra, pair.lhs), out_lhs);
                result = backend_t::InnerProductAdd(predicate, result, backend_t::Load(bra, pair.rhs), out_rhs);
            }
            return result;
        }

        const auto negative = backend_t::ExternalParity(predicate, state_base, index_context, kernel.external_z_mask);
        const auto minus_i_sine = value_t{0, -kernel.sine};
        for (const auto& scale : kernel.scales) {
            auto coefficient = backend_t::Broadcast(value_t{kernel.cosine, 0});
            coefficient = backend_t::Add(predicate, coefficient,
                                         backend_t::SignedCoefficient(minus_i_sine * scale.factor, negative));
            const auto transformed = backend_t::MultiplyDynamic(predicate, backend_t::Load(ket, scale.basis),
                                                                coefficient);
            result = backend_t::InnerProductAdd(predicate, result, backend_t::Load(bra, scale.basis), transformed);
        }
        for (const auto& pair : kernel.pairs) {
            const auto ket_lhs = backend_t::Load(ket, pair.lhs);
            const auto ket_rhs = backend_t::Load(ket, pair.rhs);
            auto out_lhs = backend_t::Multiply(predicate, ket_lhs, kernel.cosine);
            auto out_rhs = backend_t::Multiply(predicate, ket_rhs, kernel.cosine);
            out_lhs = backend_t::MultiplyAddDynamic(
                predicate, out_lhs, ket_rhs, backend_t::SignedCoefficient(minus_i_sine * pair.lhs_from_rhs, negative));
            out_rhs = backend_t::MultiplyAddDynamic(
                predicate, out_rhs, ket_lhs, backend_t::SignedCoefficient(minus_i_sine * pair.rhs_from_lhs, negative));
            result = backend_t::InnerProductAdd(predicate, result, backend_t::Load(bra, pair.lhs), out_lhs);
            result = backend_t::InnerProductAdd(predicate, result, backend_t::Load(bra, pair.rhs), out_rhs);
        }
        return result;
    }

    [[nodiscard]] static auto ExpectKernel(const DiagonalKernel<value_t>& kernel, predicate_t predicate, state_index_t,
                                           const index_context_t&, const storage_t* bra, const storage_t* ket)
        -> vector_t {
        auto result = backend_t::Zero();
        for (const auto& scale : kernel.scales) {
            const auto transformed = backend_t::MultiplyCoefficient(predicate, backend_t::Load(ket, scale.basis),
                                                                    scale.factor);
            result = backend_t::InnerProductAdd(predicate, result, backend_t::Load(bra, scale.basis), transformed);
        }
        return result;
    }

    [[nodiscard]] static auto ExpectKernel(const PairMatrixKernel<value_t>& kernel, predicate_t predicate,
                                           state_index_t, const index_context_t&, const storage_t* bra,
                                           const storage_t* ket) -> vector_t {
        auto result = backend_t::Zero();
        for (const auto& pair : kernel.pairs) {
            const auto ket_lhs = backend_t::Load(ket, pair.lhs);
            const auto ket_rhs = backend_t::Load(ket, pair.rhs);
            auto out_lhs = backend_t::Zero();
            auto out_rhs = backend_t::Zero();
            out_lhs = backend_t::MultiplyAddCoefficient(predicate, out_lhs, ket_lhs, kernel.matrix[0]);
            out_lhs = backend_t::MultiplyAddCoefficient(predicate, out_lhs, ket_rhs, kernel.matrix[1]);
            out_rhs = backend_t::MultiplyAddCoefficient(predicate, out_rhs, ket_lhs, kernel.matrix[2]);
            out_rhs = backend_t::MultiplyAddCoefficient(predicate, out_rhs, ket_rhs, kernel.matrix[3]);
            result = backend_t::InnerProductAdd(predicate, result, backend_t::Load(bra, pair.lhs), out_lhs);
            result = backend_t::InnerProductAdd(predicate, result, backend_t::Load(bra, pair.rhs), out_rhs);
        }
        return result;
    }

    [[nodiscard]] static auto ExpectKernel(const DenseMatrixKernel<value_t>& kernel, predicate_t predicate,
                                           state_index_t, const index_context_t&, const storage_t* bra,
                                           const storage_t* ket) -> vector_t {
        assert(kernel.matrix_dim != 0 && kernel.indices.size() % kernel.matrix_dim == 0);
        auto result = backend_t::Zero();
        for (size_t group = 0; group < kernel.indices.size(); group += kernel.matrix_dim) {
            for (size_t row = 0; row < kernel.matrix_dim; ++row) {
                auto transformed = backend_t::Zero();
                const auto* matrix_row = kernel.matrix.data() + row * kernel.matrix_dim;
                for (size_t column = 0; column < kernel.matrix_dim; ++column) {
                    transformed = backend_t::MultiplyAddCoefficient(
                        predicate, transformed, backend_t::Load(ket, kernel.indices[group + column]),
                        matrix_row[column]);
                }
                result = backend_t::InnerProductAdd(predicate, result,
                                                    backend_t::Load(bra, kernel.indices[group + row]), transformed);
            }
        }
        return result;
    }
};
}  // namespace mindquantum::sim::circuit_tiling::detail

#endif  // INCLUDE_SIMULATOR_CIRCUIT_TILING_DETAIL_KERNELS_H_
