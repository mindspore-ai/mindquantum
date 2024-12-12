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

#ifndef TOOLS_CUH
#define TOOLS_CUH

#include <curand_kernel.h>

#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "algorithm/qaia/csr_base.h"
#include "core/mq_base_types.h"

using mindquantum::Index;

template <typename T>
struct Bounds;

// Specialization for int8_t
template <>
struct Bounds<int8_t> {
    static __device__ int8_t UP() {
        return 127;
    }
    static __device__ int8_t LOW() {
        return -127;
    }
};

// Specialization for half
template <>
struct Bounds<half> {
    static __device__ half UP() {
        return __float2half(1.0f);
    }
    static __device__ half LOW() {
        return __float2half(-1.0f);
    }
};

// Templated CUDA kernel functions
template <typename T>
__global__ void sign_kernel(const T* __restrict__ x, T* __restrict__ signx, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    T zero = static_cast<T>(0);
    if (x[idx] > zero) {
        signx[idx] = Bounds<T>::UP();
    } else if (x[idx] < zero) {
        signx[idx] = Bounds<T>::LOW();
    } else {
        signx[idx] = zero;
    }
}

#define UP  127
#define LOW -127

__global__ void update_tail_half(const half* __restrict__ tmp, half* x, half* y, int size, float xi, float beta,
                                 float delta, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    y[idx] += static_cast<half>(dt * (xi * static_cast<float>(tmp[idx]) + beta * static_cast<float>(x[idx])));
    half x_idx = static_cast<half>(x[idx]) + y[idx] * static_cast<half>(delta * dt);
    if (x_idx >= static_cast<half>(1.0)) {
        x[idx] = 1.0;
        y[idx] = 0;
    } else if (x_idx <= static_cast<half>(-1.0)) {
        x[idx] = -1.0;
        y[idx] = 0;
    } else {
        x[idx] = x_idx;
    }
}

__global__ void update_h_tail_half(const half* __restrict__ tmp, half* x, half* y, half* h, int size, float xi,
                                   float beta, float delta, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    y[idx] += static_cast<half>(dt * (xi * static_cast<float>(tmp[idx] + h[idx]) + beta * static_cast<float>(x[idx])));
    half x_idx = static_cast<half>(x[idx]) + y[idx] * static_cast<half>(delta * dt);
    if (x_idx >= static_cast<half>(1.0)) {
        x[idx] = 1.0;
        y[idx] = 0;
    } else if (x_idx <= static_cast<half>(-1.0)) {
        x[idx] = -1.0;
        y[idx] = 0;
    } else {
        x[idx] = x_idx;
    }
}

__global__ void update_tail(const int* __restrict__ tmp, int8_t* x, int* y, int size, float xi, float beta, float delta,
                            float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    float y_new = static_cast<float>(y[idx])
                  + dt * (xi * static_cast<float>(tmp[idx] / 127) + beta * static_cast<float>(x[idx]));
    int x_idx = static_cast<int>(x[idx]) + static_cast<int>(roundf(y_new * delta * dt));
    if (x_idx >= UP) {
        x[idx] = UP;
        y[idx] = 0;
    } else if (x_idx <= LOW) {
        x[idx] = LOW;
        y[idx] = 0;
    } else {
        x[idx] = static_cast<int8_t>(x_idx);
        y[idx] = static_cast<int>(roundf(y_new));
    }
}

__global__ void update_h_tail(const int* __restrict__ tmp, int8_t* x, int* y, int* h, int size, float xi, float beta,
                              float delta, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    float y_new = static_cast<float>(y[idx])
                  + dt * (xi * static_cast<float>((tmp[idx] + h[idx]) / 127) + beta * static_cast<float>(x[idx]));
    int x_idx = static_cast<int>(x[idx]) + static_cast<int>(roundf(y_new * delta * dt));
    if (x_idx >= UP) {
        x[idx] = UP;
        y[idx] = 0;
    } else if (x_idx <= LOW) {
        x[idx] = LOW;
        y[idx] = 0;
    } else {
        x[idx] = static_cast<int8_t>(x_idx);
        y[idx] = static_cast<int>(roundf(y_new));
    }
}

// Base template (left undefined)
template <typename T>
__global__ void init_xy(T* array, int size, uint64_t seed);
// Specialization for int
template <>
__global__ void init_xy<int>(int* array, int size, uint64_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        int randomValue = curand(&state) % 7 - 3;
        array[idx] = randomValue;
    }
}
// Specialization for int8_t
template <>
__global__ void init_xy<int8_t>(int8_t* array, int size, uint64_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        int randomValue = curand(&state) % 7 - 3;
        array[idx] = static_cast<int8_t>(randomValue);
    }
}
// Specialization for half
template <>
__global__ void init_xy<half>(half* array, int size, uint64_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float rand_float = curand_uniform(&state) * 0.02f - 0.01f;
        array[idx] = __float2half(rand_float);
    }
}

template <typename T>
struct Value;

template <>
struct Value<int8_t> {
    static int8_t v(double value) {
        return static_cast<int8_t>(value * 127.0);
    }
};

template <>
struct Value<half> {
    static half v(double value) {
        return static_cast<half>(value);
    }
};

template <typename T>
void fill_J(Index* indices, Index* indptr, double* data, std::vector<T>* h_J, int N) {
    for (int i = 0; i < N; ++i) {
        Index start = indptr[i];
        Index end = indptr[i + 1];
        for (Index j = start; j < end; ++j) {
            Index col = indices[j];
            double value = data[j];
            (*h_J)[i * N + col] = Value<T>::v(value);
        }
    }
}

#endif
