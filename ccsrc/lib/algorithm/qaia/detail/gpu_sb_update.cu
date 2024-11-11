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

#include <cublas_v2.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "algorithm/qaia/csr_base.h"
#include "algorithm/qaia/detail/common.cuh"
#include "algorithm/qaia/detail/gpu_sb.cuh"
#include "algorithm/qaia/detail/para.h"
#include "algorithm/qaia/detail/tools.cuh"

using mindquantum::Index;

namespace mindquantum::algorithm::qaia::detail {

void SBBase::dSB_update_int8(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras) {
    Index* indptr = csr.indptr_;
    Index* indices = csr.indices_;
    double* data = csr.data_;
    int N = csr.dim_;
    int B = paras.B;
    float xi = paras.xi;
    float delta = paras.delta;
    float dt = paras.dt;
    int n_iter = paras.n_iter;
    int NN = N * N;
    int NB = N * B;

    std::vector<int8_t> h_J(NN, 0);
    std::vector<int8_t> h_x(NB);
    std::vector<int> h_y(NB);

    fill_J<int8_t>(indices, indptr, data, h_J, N);

    int8_t *d_J, *d_x, *signx;
    int *d_y, *tmp;
    HANDLE_ERROR(cudaMalloc((void**) &d_J, NN * sizeof(int8_t)));
    HANDLE_ERROR(cudaMalloc((void**) &d_x, NB * sizeof(int8_t)));
    HANDLE_ERROR(cudaMalloc((void**) &d_y, NB * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &tmp, NB * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &signx, NB * sizeof(int8_t)));
    HANDLE_ERROR(cudaMemcpy(d_J, h_J.data(), NN * sizeof(int8_t), cudaMemcpyHostToDevice));

    init_xy<int8_t><<<(NB + 255) / 256, 256>>>(d_x, NB, time(NULL));
    init_xy<int><<<(NB + 255) / 256, 256>>>(d_y, NB, time(NULL));

    cublasHandle_t handle;
    cublasCreate(&handle);
    int alpha = 1;
    int b = 0;
    for (int i = 0; i < n_iter; i++) {
        float beta = (n_iter == 1) ? -delta : static_cast<float>(i) / (n_iter - 1) - delta;

        sign_kernel<int8_t><<<(NB + 255) / 256, 256>>>(d_x, signx, NB);

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, B, N, &alpha, d_J, CUDA_R_8I, N, signx,
                                  CUDA_R_8I, N, &b, tmp, CUDA_R_32I, N, CUDA_R_32I, CUBLAS_GEMM_ALGO15_TENSOR_OP));

        update_tail<<<(NB + 255) / 256, 256>>>(tmp, d_x, d_y, NB, xi, beta, delta, dt);
    }
    CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, NB * sizeof(int8_t), cudaMemcpyDeviceToHost))

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < B; j++) {
            x[i * B + j] = static_cast<double>(h_x[j * N + i]) / 127.0;
        }
    }

    HANDLE_ERROR(cudaFree(d_J));
    HANDLE_ERROR(cudaFree(d_x));
    HANDLE_ERROR(cudaFree(d_y));
    HANDLE_ERROR(cudaFree(tmp));
    HANDLE_ERROR(cudaFree(signx));
}

void SBBase::bSB_update_int8(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras) {
    Index* indptr = csr.indptr_;
    Index* indices = csr.indices_;
    double* data = csr.data_;
    int N = csr.dim_;
    int B = paras.B;
    float xi = paras.xi;
    float delta = paras.delta;
    float dt = paras.dt;
    int n_iter = paras.n_iter;
    int NN = N * N;
    int NB = N * B;

    std::vector<int8_t> h_J(NN, 0);
    std::vector<int8_t> h_x(NB);
    std::vector<int> h_y(NB);

    fill_J<int8_t>(indices, indptr, data, h_J, N);

    int8_t *d_J, *d_x;
    int *d_y, *tmp;
    HANDLE_ERROR(cudaMalloc((void**) &d_J, NN * sizeof(int8_t)));
    HANDLE_ERROR(cudaMalloc((void**) &d_x, NB * sizeof(int8_t)));
    HANDLE_ERROR(cudaMalloc((void**) &d_y, NB * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &tmp, NB * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_J, h_J.data(), NN * sizeof(int8_t), cudaMemcpyHostToDevice));

    init_xy<int8_t><<<(NB + 255) / 256, 256>>>(d_x, NB, time(NULL));
    init_xy<int><<<(NB + 255) / 256, 256>>>(d_y, NB, time(NULL));

    cublasHandle_t handle;
    cublasCreate(&handle);
    int alpha = 1;
    int b = 0;
    for (int i = 0; i < n_iter; i++) {
        float beta = (n_iter == 1) ? -delta : static_cast<float>(i) / (n_iter - 1) - delta;

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, B, N, &alpha, d_J, CUDA_R_8I, N, d_x, CUDA_R_8I,
                                  N, &b, tmp, CUDA_R_32I, N, CUDA_R_32I, CUBLAS_GEMM_ALGO15_TENSOR_OP));

        update_tail<<<(NB + 255) / 256, 256>>>(tmp, d_x, d_y, NB, xi, beta, delta, dt);
    }
    CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, NB * sizeof(int8_t), cudaMemcpyDeviceToHost))

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < B; j++) {
            x[i * B + j] = static_cast<double>(h_x[j * N + i]) / 127.0;
        }
    }
    HANDLE_ERROR(cudaFree(d_J));
    HANDLE_ERROR(cudaFree(d_x));
    HANDLE_ERROR(cudaFree(d_y));
    HANDLE_ERROR(cudaFree(tmp));
}

void SBBase::bSB_update_fp16(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras) {
    Index* indptr = csr.indptr_;
    Index* indices = csr.indices_;
    double* data = csr.data_;
    int N = csr.dim_;
    int B = paras.B;
    float xi = paras.xi;
    float delta = paras.delta;
    float dt = paras.dt;
    int n_iter = paras.n_iter;
    int NN = N * N;
    int NB = N * B;

    std::vector<half> h_J(NN, 0);
    std::vector<half> h_x(NB);
    std::vector<half> h_y(NB);

    fill_J<half>(indices, indptr, data, h_J, N);

    half *d_J, *d_x;
    half *d_y, *tmp;
    HANDLE_ERROR(cudaMalloc((void**) &d_J, NN * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &d_x, NB * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &d_y, NB * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &tmp, NB * sizeof(half)));
    HANDLE_ERROR(cudaMemcpy(d_J, h_J.data(), NN * sizeof(half), cudaMemcpyHostToDevice));

    init_xy<half><<<(NB + 255) / 256, 256>>>(d_x, NB, time(NULL));
    init_xy<half><<<(NB + 255) / 256, 256>>>(d_y, NB, time(NULL));

    cublasHandle_t handle;
    cublasCreate(&handle);
    // cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    half b = 0.0;
    half alpha = 1.0;
    for (int i = 0; i < n_iter; i++) {
        float beta = (n_iter == 1) ? -delta : static_cast<float>(i) / (n_iter - 1) - delta;

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, B, N, N, &alpha, d_x, CUDA_R_16F, B, d_J,
                                  CUDA_R_16F, N, &b, tmp, CUDA_R_16F, B, CUBLAS_COMPUTE_16F,
                                  CUBLAS_GEMM_ALGO15_TENSOR_OP));

        update_tail_half<<<(NB + 255) / 256, 256>>>(tmp, d_x, d_y, NB, xi, beta, delta, dt);
    }
    CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, NB * sizeof(half), cudaMemcpyDeviceToHost))

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < B; j++) {
            x[i * B + j] = static_cast<double>(h_x[i * B + j]);
        }
    }

    HANDLE_ERROR(cudaFree(d_J));
    HANDLE_ERROR(cudaFree(d_x));
    HANDLE_ERROR(cudaFree(d_y));
    HANDLE_ERROR(cudaFree(tmp));
}

void SBBase::dSB_update_fp16(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras) {
    Index* indptr = csr.indptr_;
    Index* indices = csr.indices_;
    double* data = csr.data_;
    int N = csr.dim_;
    int B = paras.B;
    float xi = paras.xi;
    float delta = paras.delta;
    float dt = paras.dt;
    int n_iter = paras.n_iter;
    int NN = N * N;
    int NB = N * B;

    std::vector<half> h_J(NN, 0);
    std::vector<half> h_x(NB);
    std::vector<half> h_y(NB);

    fill_J<half>(indices, indptr, data, h_J, N);

    half *d_J, *d_x;
    half *d_y, *tmp, *signx;
    HANDLE_ERROR(cudaMalloc((void**) &d_J, NN * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &d_x, NB * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &d_y, NB * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &tmp, NB * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &signx, NB * sizeof(half)));
    HANDLE_ERROR(cudaMemcpy(d_J, h_J.data(), NN * sizeof(half), cudaMemcpyHostToDevice));

    init_xy<half><<<(NB + 255) / 256, 256>>>(d_x, NB, time(NULL));
    init_xy<half><<<(NB + 255) / 256, 256>>>(d_y, NB, time(NULL));

    cublasHandle_t handle;
    cublasCreate(&handle);
    half b = 0.0;
    half alpha = 1.0;
    for (int i = 0; i < n_iter; i++) {
        float beta = (n_iter == 1) ? -delta : static_cast<float>(i) / (n_iter - 1) - delta;

        sign_kernel<half><<<(NB + 255) / 256, 256>>>(d_x, signx, NB);

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, B, N, N, &alpha, signx, CUDA_R_16F, B, d_J,
                                  CUDA_R_16F, N, &b, tmp, CUDA_R_16F, B, CUBLAS_COMPUTE_16F,
                                  CUBLAS_GEMM_ALGO15_TENSOR_OP));

        update_tail_half<<<(NB + 255) / 256, 256>>>(tmp, d_x, d_y, NB, xi, beta, delta, dt);
    }
    CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, NB * sizeof(half), cudaMemcpyDeviceToHost))

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < B; j++) {
            x[i * B + j] = static_cast<double>(h_x[i * B + j]);
        }
    }
    HANDLE_ERROR(cudaFree(d_J));
    HANDLE_ERROR(cudaFree(d_x));
    HANDLE_ERROR(cudaFree(d_y));
    HANDLE_ERROR(cudaFree(tmp));
    HANDLE_ERROR(cudaFree(signx));
}

void SBBase::dSB_update_h_int8(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras, double* h, int ndim) {
    Index* indptr = csr.indptr_;
    Index* indices = csr.indices_;
    double* data = csr.data_;
    int N = csr.dim_;
    int B = paras.B;
    float xi = paras.xi;
    float delta = paras.delta;
    float dt = paras.dt;
    int n_iter = paras.n_iter;
    int NN = N * N;
    int NB = N * B;

    std::vector<int8_t> h_J(NN, 0);
    std::vector<int8_t> h_x(NB);
    std::vector<int> h_y(NB);

    fill_J<int8_t>(indices, indptr, data, h_J, N);

    int8_t *d_J, *d_x, *signx;
    int *d_y, *tmp, *d_h;
    HANDLE_ERROR(cudaMalloc((void**) &d_J, NN * sizeof(int8_t)));
    HANDLE_ERROR(cudaMalloc((void**) &d_x, NB * sizeof(int8_t)));
    HANDLE_ERROR(cudaMalloc((void**) &d_y, NB * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &tmp, NB * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &signx, NB * sizeof(int8_t)));
    HANDLE_ERROR(cudaMemcpy(d_J, h_J.data(), NN * sizeof(int8_t), cudaMemcpyHostToDevice));
    std::vector<int> h_h(NB, 0);
    for (int i = 0; i < NB; i++) {
        h_h[i] = static_cast<int>(h[i] * 127);
    }
    HANDLE_ERROR(cudaMalloc((void**) &d_h, NB * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_h, h_h.data(), NB * sizeof(int), cudaMemcpyHostToDevice));

    init_xy<int8_t><<<(NB + 255) / 256, 256>>>(d_x, NB, time(NULL));
    init_xy<int><<<(NB + 255) / 256, 256>>>(d_y, NB, time(NULL));

    cublasHandle_t handle;
    cublasCreate(&handle);
    int alpha = 1;
    int b = 0;
    for (int i = 0; i < n_iter; i++) {
        float beta = (n_iter == 1) ? -delta : static_cast<float>(i) / (n_iter - 1) - delta;

        sign_kernel<int8_t><<<(NB + 255) / 256, 256>>>(d_x, signx, NB);

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, B, N, &alpha, d_J, CUDA_R_8I, N, signx,
                                  CUDA_R_8I, N, &b, tmp, CUDA_R_32I, N, CUDA_R_32I, CUBLAS_GEMM_ALGO15_TENSOR_OP));

        update_h_tail<<<(NB + 255) / 256, 256>>>(tmp, d_x, d_y, d_h, NB, xi, beta, delta, dt);
    }
    CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, NB * sizeof(int8_t), cudaMemcpyDeviceToHost))

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < B; j++) {
            x[i * B + j] = static_cast<double>(h_x[j * N + i]) / 127.0;
        }
    }
    HANDLE_ERROR(cudaFree(d_h));
    HANDLE_ERROR(cudaFree(d_J));
    HANDLE_ERROR(cudaFree(d_x));
    HANDLE_ERROR(cudaFree(d_y));
    HANDLE_ERROR(cudaFree(tmp));
    HANDLE_ERROR(cudaFree(signx));
}

void SBBase::bSB_update_h_int8(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras, double* h, int ndim) {
    Index* indptr = csr.indptr_;
    Index* indices = csr.indices_;
    double* data = csr.data_;
    int N = csr.dim_;
    int B = paras.B;
    float xi = paras.xi;
    float delta = paras.delta;
    float dt = paras.dt;
    int n_iter = paras.n_iter;
    int NN = N * N;
    int NB = N * B;

    std::vector<int8_t> h_J(NN, 0);
    std::vector<int8_t> h_x(NB);
    std::vector<int> h_y(NB);

    fill_J<int8_t>(indices, indptr, data, h_J, N);

    int8_t *d_J, *d_x;
    int *d_y, *tmp, *d_h;
    HANDLE_ERROR(cudaMalloc((void**) &d_J, NN * sizeof(int8_t)));
    HANDLE_ERROR(cudaMalloc((void**) &d_x, NB * sizeof(int8_t)));
    HANDLE_ERROR(cudaMalloc((void**) &d_y, NB * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**) &tmp, NB * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_J, h_J.data(), NN * sizeof(int8_t), cudaMemcpyHostToDevice));
    std::vector<int> h_h(NB, 0);
    for (int i = 0; i < NB; i++) {
        h_h[i] = static_cast<int>(data[i] * 127);
    }
    HANDLE_ERROR(cudaMalloc((void**) &d_h, NB * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_h, h_h.data(), NB * sizeof(int), cudaMemcpyHostToDevice));

    init_xy<int8_t><<<(NB + 255) / 256, 256>>>(d_x, NB, time(NULL));
    init_xy<int><<<(NB + 255) / 256, 256>>>(d_y, NB, time(NULL));

    cublasHandle_t handle;
    cublasCreate(&handle);
    int alpha = 1;
    int b = 0;
    for (int i = 0; i < n_iter; i++) {
        float beta = (n_iter == 1) ? -delta : static_cast<float>(i) / (n_iter - 1) - delta;

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, B, N, &alpha, d_J, CUDA_R_8I, N, d_x, CUDA_R_8I,
                                  N, &b, tmp, CUDA_R_32I, N, CUDA_R_32I, CUBLAS_GEMM_ALGO15_TENSOR_OP));

        update_h_tail<<<(NB + 255) / 256, 256>>>(tmp, d_x, d_y, d_h, NB, xi, beta, delta, dt);
    }
    CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, NB * sizeof(int8_t), cudaMemcpyDeviceToHost))

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < B; j++) {
            x[i * B + j] = static_cast<double>(h_x[j * N + i]) / 127.0;
        }
    }

    HANDLE_ERROR(cudaFree(d_h));
    HANDLE_ERROR(cudaFree(d_J));
    HANDLE_ERROR(cudaFree(d_x));
    HANDLE_ERROR(cudaFree(d_y));
    HANDLE_ERROR(cudaFree(tmp));
}

void SBBase::bSB_update_h_fp16(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras, double* h, int ndim) {
    Index* indptr = csr.indptr_;
    Index* indices = csr.indices_;
    double* data = csr.data_;
    int N = csr.dim_;
    int B = paras.B;
    float xi = paras.xi;
    float delta = paras.delta;
    float dt = paras.dt;
    int n_iter = paras.n_iter;
    int NN = N * N;
    int NB = N * B;

    std::vector<half> h_J(NN, 0);
    std::vector<half> h_x(NB);
    std::vector<half> h_y(NB);

    fill_J<half>(indices, indptr, data, h_J, N);

    half *d_J, *d_x;
    half *d_y, *tmp, *d_h;
    HANDLE_ERROR(cudaMalloc((void**) &d_J, NN * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &d_x, NB * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &d_y, NB * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &tmp, NB * sizeof(half)));
    HANDLE_ERROR(cudaMemcpy(d_J, h_J.data(), NN * sizeof(half), cudaMemcpyHostToDevice));
    std::vector<half> h_h(NB, 0);
    for (int i = 0; i < NB; i++) {
        h_h[i] = static_cast<half>(h[i]);
    }
    HANDLE_ERROR(cudaMalloc((void**) &d_h, NB * sizeof(half)));
    HANDLE_ERROR(cudaMemcpy(d_h, h_h.data(), NB * sizeof(half), cudaMemcpyHostToDevice));

    init_xy<half><<<(NB + 255) / 256, 256>>>(d_x, NB, time(NULL));
    init_xy<half><<<(NB + 255) / 256, 256>>>(d_y, NB, time(NULL));

    cublasHandle_t handle;
    cublasCreate(&handle);
    // cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    half b = 0.0;
    half alpha = 1.0;
    for (int i = 0; i < n_iter; i++) {
        float beta = (n_iter == 1) ? -delta : static_cast<float>(i) / (n_iter - 1) - delta;

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, B, N, N, &alpha, d_x, CUDA_R_16F, B, d_J,
                                  CUDA_R_16F, N, &b, tmp, CUDA_R_16F, B, CUBLAS_COMPUTE_16F,
                                  CUBLAS_GEMM_ALGO15_TENSOR_OP));

        update_h_tail_half<<<(NB + 255) / 256, 256>>>(tmp, d_x, d_y, d_h, NB, xi, beta, delta, dt);
    }
    CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, NB * sizeof(half), cudaMemcpyDeviceToHost))

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < B; j++) {
            x[i * B + j] = static_cast<double>(h_x[i * B + j]);
        }
    }

    HANDLE_ERROR(cudaFree(d_h));
    HANDLE_ERROR(cudaFree(d_J));
    HANDLE_ERROR(cudaFree(d_x));
    HANDLE_ERROR(cudaFree(d_y));
    HANDLE_ERROR(cudaFree(tmp));
}

void SBBase::dSB_update_h_fp16(mindquantum::sparse::CsrBase<double> csr, double* x, Para paras, double* h, int ndim) {
    Index* indptr = csr.indptr_;
    Index* indices = csr.indices_;
    double* data = csr.data_;
    int N = csr.dim_;
    int B = paras.B;
    float xi = paras.xi;
    float delta = paras.delta;
    float dt = paras.dt;
    int n_iter = paras.n_iter;
    int NN = N * N;
    int NB = N * B;

    std::vector<half> h_J(NN, 0);
    std::vector<half> h_x(NB);
    std::vector<half> h_y(NB);

    fill_J<half>(indices, indptr, data, h_J, N);

    half *d_J, *d_x;
    half *d_y, *tmp, *d_h, *signx;
    HANDLE_ERROR(cudaMalloc((void**) &d_J, NN * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &d_x, NB * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &d_y, NB * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &tmp, NB * sizeof(half)));
    HANDLE_ERROR(cudaMalloc((void**) &signx, NB * sizeof(half)));
    HANDLE_ERROR(cudaMemcpy(d_J, h_J.data(), NN * sizeof(half), cudaMemcpyHostToDevice));
    std::vector<half> h_h(NB, 0);
    for (int i = 0; i < NB; i++) {
        h_h[i] = static_cast<half>(h[i]);
    }
    HANDLE_ERROR(cudaMalloc((void**) &d_h, NB * sizeof(half)));
    HANDLE_ERROR(cudaMemcpy(d_h, h_h.data(), NB * sizeof(half), cudaMemcpyHostToDevice));

    init_xy<half><<<(NB + 255) / 256, 256>>>(d_x, NB, time(NULL));
    init_xy<half><<<(NB + 255) / 256, 256>>>(d_y, NB, time(NULL));

    cublasHandle_t handle;
    cublasCreate(&handle);
    half b = 0.0;
    half alpha = 1.0;
    for (int i = 0; i < n_iter; i++) {
        float beta = (n_iter == 1) ? -delta : static_cast<float>(i) / (n_iter - 1) - delta;

        sign_kernel<half><<<(NB + 255) / 256, 256>>>(d_x, signx, NB);

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, B, N, N, &alpha, signx, CUDA_R_16F, B, d_J,
                                  CUDA_R_16F, N, &b, tmp, CUDA_R_16F, B, CUBLAS_COMPUTE_16F,
                                  CUBLAS_GEMM_ALGO15_TENSOR_OP));

        update_h_tail_half<<<(NB + 255) / 256, 256>>>(tmp, d_x, d_y, d_h, NB, xi, beta, delta, dt);
    }
    CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, NB * sizeof(half), cudaMemcpyDeviceToHost))

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < B; j++) {
            x[i * B + j] = static_cast<double>(h_x[i * B + j]);
        }
    }

    HANDLE_ERROR(cudaFree(d_h));
    HANDLE_ERROR(cudaFree(d_J));
    HANDLE_ERROR(cudaFree(d_x));
    HANDLE_ERROR(cudaFree(d_y));
    HANDLE_ERROR(cudaFree(tmp));
    HANDLE_ERROR(cudaFree(signx));
}

void SBBase::cublas_warmup(int N, int B) {
    int8_t *d_J, *d_x;
    int* tmp;
    HANDLE_ERROR(cudaMalloc((void**) &d_J, N * N * sizeof(int8_t)));
    HANDLE_ERROR(cudaMalloc((void**) &d_x, N * B * sizeof(int8_t)));
    HANDLE_ERROR(cudaMalloc((void**) &tmp, N * B * sizeof(int)));
    cublasHandle_t handle;
    cublasCreate(&handle);
    int alpha = 1;
    int beta = 0;
    for (int i = 0; i < 100; i++) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, B, N, &alpha, d_J, CUDA_R_8I, N, d_x, CUDA_R_8I,
                                  N, &beta, tmp, CUDA_R_32I, N, CUDA_R_32I, CUBLAS_GEMM_ALGO15_TENSOR_OP));
    }
    HANDLE_ERROR(cudaFree(d_J));
    HANDLE_ERROR(cudaFree(d_x));
    HANDLE_ERROR(cudaFree(tmp));
}

}  // namespace mindquantum::algorithm::qaia::detail