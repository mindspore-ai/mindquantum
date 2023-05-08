//   Copyright 2022 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
#include <limits>

#include <thrust/transform_reduce.h>

#include "config/openmp.hpp"

#include "simulator/utils.hpp"
#include "simulator/vector/detail/gpu_vector_double_policy.cuh"
#include "simulator/vector/detail/gpu_vector_float_policy.cuh"
#include "simulator/vector/detail/gpu_vector_policy.cuh"
#include "thrust/device_ptr.h"
#include "thrust/device_vector.h"
#include "thrust/functional.h"
#include "thrust/inner_product.h"

namespace mindquantum::sim::vector::detail {
template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    qs_data_p_t qs;
    cudaMalloc((void**) &qs, sizeof(qs_data_t) * dim);  // NOLINT
    cudaMemset(qs, 0, sizeof(qs_data_t) * dim);
    if (zero_state) {
        qs_data_t one = qs_data_t(1.0, 0.0);
        cudaMemcpy(qs, &one, sizeof(qs_data_t), cudaMemcpyHostToDevice);
    }
    return qs;
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::FreeState(qs_data_p_t* qs_p) {
    auto& qs = (*qs_p);
    if (qs != nullptr) {
        cudaFree(qs);
        qs = nullptr;
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::Reset(qs_data_p_t* qs_p, index_t dim) {
    derived::FreeState(qs_p);
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::Display(const qs_data_p_t& qs, qbit_t n_qubits, qbit_t q_limit) {
    if (n_qubits > q_limit) {
        n_qubits = q_limit;
    }
    std::cout << n_qubits << " qubits gpu simulator (little endian)." << std::endl;
    if (qs == nullptr) {
        std::cout << "(" << 1 << ", " << 0 << ")" << std::endl;
        for (index_t i = 0; i < (1UL << n_qubits) - 1; i++) {
            std::cout << "(" << 0 << ", " << 0 << ")" << std::endl;
        }
    } else {
        std::complex<calc_type>* h_qs = reinterpret_cast<std::complex<calc_type>*>(
            malloc((1UL << n_qubits) * sizeof(std::complex<calc_type>)));
        cudaMemcpy(h_qs, qs, sizeof(qs_data_t) * (1UL << n_qubits), cudaMemcpyDeviceToHost);
        for (index_t i = 0; i < (1UL << n_qubits); i++) {
            std::cout << "(" << h_qs[i].real() << ", " << h_qs[i].imag() << ")" << std::endl;
        }
        free(h_qs);
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::SetToZeroExcept(qs_data_p_t* qs_p, index_t ctrl_mask, index_t dim) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    thrust::counting_iterator<index_t> i(0);
    thrust::for_each(i, i + dim, [=] __device__(index_t i) {
        if ((i & ctrl_mask) != ctrl_mask) {
            qs[i] = 0;
        }
    });
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::GetQS(const qs_data_p_t& qs, index_t dim) -> py_qs_datas_t {
    py_qs_datas_t out(dim);
    if (qs == nullptr) {
        out[0] = 1.0;
    } else {
        cudaMemcpy(out.data(), qs, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToHost);
    }
    return out;
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::SetQS(qs_data_p_t* qs_p, const py_qs_datas_t& qs_out, index_t dim) {
    auto& qs = (*qs_p);
    if (qs_out.size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    if (qs == nullptr) {
        qs = derived::InitState(dim, false);
    }
    cudaMemcpy(qs, qs_out.data(), sizeof(qs_data_t) * dim, cudaMemcpyHostToDevice);
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectationOfTerms(qs_data_p_t* bra_p, qs_data_p_t* ket_p,
                                                                   const std::vector<PauliTerm<calc_type>>& ham,
                                                                   index_t dim) -> py_qs_data_t {
    auto& bra = (*bra_p);
    auto& ket = (*ket_p);
    if (bra == nullptr) {
        bra = derived::InitState(dim);
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
    }
    qs_data_t out = 0.0;
    for (const auto& [pauli_string, coeff] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        auto mask_z = mask.mask_z;
        auto mask_y = mask.mask_y;
        auto num_y = mask.num_y;
        auto this_coeff = qs_data_t(coeff);
        thrust::counting_iterator<index_t> i(0);
        out += thrust::transform_reduce(
            i, i + dim,
            [=] __device__(index_t i) {
                auto j = (i ^ mask_f);
                qs_data_t tmp = 0.0;
                if (i <= j) {
                    auto axis2power = __popcll(i & mask_z);
                    auto axis3power = __popcll(i & mask_y);
                    auto idx = (num_y + 2 * axis3power + 2 * axis2power) & 3;
                    auto c = qs_data_t(1, 0);
                    if (idx == 1) {
                        c = qs_data_t(0, 1);
                    } else if (idx == 2) {
                        c = qs_data_t(-1, 0);
                    } else if (idx == 3) {
                        c = qs_data_t(0, -1);
                    }
                    tmp += thrust::conj(bra[j]) * ket[i] * this_coeff * c;
                    if (i != j) {
                        tmp += thrust::conj(bra[i]) * ket[j] * this_coeff / c;
                    }
                }
                return tmp;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    return out;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ApplyTerms(qs_data_p_t* qs_p,
                                                           const std::vector<PauliTerm<calc_type>>& ham, index_t dim)
    -> qs_data_p_t {
    auto& qs = (*qs_p);
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    qs_data_p_t out = derived::InitState(dim, false);
    for (const auto& [pauli_string, coeff] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        if (dim >= (1UL << 18)) {
            auto mask_z = mask.mask_z;
            auto mask_y = mask.mask_y;
            auto num_y = mask.num_y;
            auto this_coeff = qs_data_t(coeff);
            thrust::counting_iterator<index_t> i(0);
            thrust::for_each(i, i + dim, [=] __device__(index_t i) {
                auto j = (i ^ mask_f);
                if (i <= j) {
                    auto axis2power = __popcll(i & mask_z);
                    auto axis3power = __popcll(i & mask_y);
                    auto idx = (num_y + 2 * axis3power + 2 * axis2power) & 3;
                    auto c = qs_data_t(1, 0);
                    if (idx == 1) {
                        c = qs_data_t(0, 1);
                    } else if (idx == 2) {
                        c = qs_data_t(-1, 0);
                    } else if (idx == 3) {
                        c = qs_data_t(0, -1);
                    }
                    out[j] += qs[i] * this_coeff * c;
                    if (i != j) {
                        out[i] += qs[j] * this_coeff / c;
                    }
                }
            });
        } else {
            ApplyTerm<qs_data_t, qs_data_p_t, calc_type>
                <<<128, 128>>>(out, qs, coeff, mask.num_y, mask.mask_y, mask.mask_z, mask_f, dim);
        }
    }
    return out;
};

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::GroundStateOfZZs(const std::map<index_t, calc_type>& masks_value,
                                                                 qbit_t n_qubits) -> calc_type {
    auto n_mask = masks_value.size();
    thrust::device_vector<index_t> mask_device;
    thrust::device_vector<calc_type> value_device;
    for (auto& [mask, value] : masks_value) {
        mask_device.push_back(mask);
        value_device.push_back(value);
    }
    auto mask_ptr = thrust::raw_pointer_cast(mask_device.data());
    auto value_ptr = thrust::raw_pointer_cast(value_device.data());
    thrust::counting_iterator<size_t> l(0);

    auto res = thrust::transform_reduce(
        l, l + (1UL << n_qubits),
        [=] __device__(size_t l) {
            calc_type ith_energy = 0;
            for (int i = 0; i < n_mask; i++) {
                if (__popcll(l & mask_ptr[i]) & 1) {
                    ith_energy -= value_ptr[i];
                } else {
                    ith_energy += value_ptr[i];
                }
            }
            return ith_energy;
        },
        std::numeric_limits<calc_type>::max(), thrust::minimum<calc_type>());
    return res;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::Copy(qs_data_p_t qs, index_t dim) -> qs_data_p_t {
    qs_data_p_t out = nullptr;
    if (qs != nullptr) {
        cudaMalloc((void**) &out, sizeof(qs_data_t) * dim);  // NOLINT
        cudaMemcpy(out, qs, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToDevice);
    }
    return out;
};

template struct GPUVectorPolicyBase<GPUVectorPolicyFloat, float>;
template struct GPUVectorPolicyBase<GPUVectorPolicyDouble, double>;

}  // namespace mindquantum::sim::vector::detail
