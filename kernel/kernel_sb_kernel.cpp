/**
 * @file kernel_sb_kernel.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

class KernelSbKernel
{
public:
    __aicore__ inline KernelSbKernel() {}
    __aicore__ inline void Init(GM_ADDR J, GM_ADDR x, GM_ADDR y, GM_ADDR p, GM_ADDR h, int32_t has_h,
                                float delta, float xi, float dt, int32_t niter, int32_t n)
    {
        this->delta = delta;
        this->xi = xi;
        this->dt = dt;
        this->niter = niter;
        this->n = n;
        // Vector operators (e.g. Mul/ReduceSum) require the length to be aligned.
        // Keep the math defined on the original `n`, and pad the tail with zeros.
        constexpr int32_t kAlign = 16;
        this->n_aligned = ((n + kAlign - 1) / kAlign) * kAlign;
        constexpr int32_t kTileElems = 1024;
        this->tile_elems = (n_aligned < kTileElems) ? n_aligned : kTileElems;
        this->has_h = has_h;

        // Global Buffers
        // J is N x N
        J_gm.SetGlobalBuffer((__gm__ float *)J, n * n);
        // x and y are N (for this batch item)
        x_gm.SetGlobalBuffer((__gm__ float *)x, n);
        y_gm.SetGlobalBuffer((__gm__ float *)y, n);
        // p is niter
        p_gm.SetGlobalBuffer((__gm__ float *)p, niter);
        if (this->has_h != 0) {
            // h is N
            h_gm.SetGlobalBuffer((__gm__ float *)h, n);
        }

        // Allocate UB sized to the runtime problem size.
        // Note: UB capacity is limited; this kernel assumes `n` is small enough
        // to fit the required buffers in UB.
        uint32_t buf_size = static_cast<uint32_t>(n_aligned) * sizeof(float);
        uint32_t tile_buf_size = static_cast<uint32_t>(tile_elems) * sizeof(float);
        uint32_t reduce_out_size = tile_buf_size;
        uint32_t mask_size = static_cast<uint32_t>((n_aligned + 7) / 8) * sizeof(uint8_t);

        pipe.InitBuffer(ub_x, 1, buf_size);
        pipe.InitBuffer(ub_y, 1, buf_size);
        pipe.InitBuffer(ub_sx, 1, buf_size);
        pipe.InitBuffer(ub_tmp, 1, buf_size);
        pipe.InitBuffer(ub_J_row0, 1, tile_buf_size);
        pipe.InitBuffer(ub_J_row1, 1, tile_buf_size);
        pipe.InitBuffer(ub_mv_res, 1, buf_size);
        pipe.InitBuffer(ub_work, 1, buf_size);
        pipe.InitBuffer(ub_h, 1, buf_size);
        pipe.InitBuffer(ub_reduce_out, 1, reduce_out_size);
        pipe.InitBuffer(ub_mask, 1, mask_size);
    }

    __aicore__ inline void Process()
    {
        // Load x and y
        AscendC::LocalTensor<float> x_local = ub_x.AllocTensor<float>();
        AscendC::LocalTensor<float> y_local = ub_y.AllocTensor<float>();

        AscendC::DataCopy(x_local, x_gm, n);
        AscendC::DataCopy(y_local, y_gm, n);
        event_t eventIdMte2ToVXY = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVXY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVXY);
        if (n_aligned > n) {
            AscendC::Duplicate(x_local[n], 0.0f, n_aligned - n);
            AscendC::Duplicate(y_local[n], 0.0f, n_aligned - n);
        }

        AscendC::LocalTensor<float> sx_local = ub_sx.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp_local = ub_tmp.AllocTensor<float>();
        AscendC::LocalTensor<float> J_row0 = ub_J_row0.AllocTensor<float>();
        AscendC::LocalTensor<float> J_row1 = ub_J_row1.AllocTensor<float>();
        AscendC::LocalTensor<float> mv_res_local = ub_mv_res.AllocTensor<float>();
        AscendC::LocalTensor<float> work_local = ub_work.AllocTensor<float>();
        AscendC::LocalTensor<float> h_local = ub_h.AllocTensor<float>();
        AscendC::LocalTensor<float> reduce_out_local = ub_reduce_out.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> mask_local = ub_mask.AllocTensor<uint8_t>();

        if (has_h != 0) {
            AscendC::DataCopy(h_local, h_gm, n);
            event_t eventIdMte2ToVH = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVH);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVH);
            if (n_aligned > n) {
                AscendC::Duplicate(h_local[n], 0.0f, n_aligned - n);
            }
        } else {
            AscendC::Duplicate(h_local, 0.0f, n_aligned);
        }

        for (int32_t iter = 0; iter < niter; ++iter)
        {
            // Read p[iter]
            float p_val = p_gm.GetValue(iter);

            // 1. Calculate Sign(x)
            // Match numpy.sign: -1 for x<0, 0 for x==0, 1 for x>0.
            AscendC::Sign(sx_local, x_local, n_aligned);

            // 2. Calculate J.dot(sign(x))
            AscendC::Duplicate(mv_res_local, 0.0f, n_aligned);
            event_t eventIdVToS = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
            if (n <= tile_elems) {
                const int32_t chunk = n;
                event_t eventIdMte2ToV0 = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                event_t eventIdMte2ToV1 = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                AscendC::DataCopy(J_row0, J_gm[0], chunk);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV0);

                int32_t cur_buf = 0;
                for (int32_t i = 0; i < n; ++i) {
                    event_t cur_event = (cur_buf == 0) ? eventIdMte2ToV0 : eventIdMte2ToV1;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(cur_event);

                    int32_t next_buf = cur_buf ^ 1;
                    if (i + 1 < n) {
                        if (next_buf == 0) {
                            AscendC::DataCopy(J_row0, J_gm[(i + 1) * n], chunk);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV0);
                        } else {
                            AscendC::DataCopy(J_row1, J_gm[(i + 1) * n], chunk);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV1);
                        }
                    }

                    AscendC::LocalTensor<float> J_row = (cur_buf == 0) ? J_row0 : J_row1;
                    if (chunk < tile_elems) {
                        AscendC::Duplicate(J_row[chunk], 0.0f, tile_elems - chunk);
                    }
                    AscendC::Mul(tmp_local, J_row, sx_local, tile_elems);
                    AscendC::ReduceSum(reduce_out_local, tmp_local, work_local, tile_elems);
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
                    mv_res_local.SetValue(i, reduce_out_local.GetValue(0));

                    cur_buf = next_buf;
                }
            } else {
                event_t eventIdMte2ToV = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                for (int32_t i = 0; i < n; ++i)
                {
                    float acc = 0.0f;
                    for (int32_t k = 0; k < n; k += tile_elems) {
                        int32_t chunk = n - k;
                        if (chunk > tile_elems) {
                            chunk = tile_elems;
                        }

                        AscendC::DataCopy(J_row0, J_gm[i * n + k], chunk);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                        if (chunk < tile_elems) {
                            AscendC::Duplicate(J_row0[chunk], 0.0f, tile_elems - chunk);
                        }

                        AscendC::Mul(tmp_local, J_row0, sx_local[k], tile_elems);
                        AscendC::ReduceSum(reduce_out_local, tmp_local, work_local, tile_elems);
                        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
                        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
                        acc += reduce_out_local.GetValue(0);
                    }
                    mv_res_local.SetValue(i, acc);
                }
            }
            event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);

            // 3. Update y
            // Term 1: (-delta - p[i]) * x -> (-delta + p_val) * x
            float c1 = -delta + p_val;
            AscendC::Muls(tmp_local, x_local, c1, n_aligned);

            // Term 2: xi * mv_res
            if (has_h != 0) {
                AscendC::Add(work_local, mv_res_local, h_local, n_aligned);
                AscendC::Muls(work_local, work_local, xi, n_aligned);
            } else {
                AscendC::Muls(work_local, mv_res_local, xi, n_aligned);
            }

            // Sum terms
            AscendC::Add(tmp_local, tmp_local, work_local, n_aligned);

            // Multiply by dt
            AscendC::Muls(tmp_local, tmp_local, dt, n_aligned);

            // Update y
            AscendC::Add(y_local, y_local, tmp_local, n_aligned);

            // 4. Update x
            // x = x + dt * y * delta
            float c2 = dt * delta;
            AscendC::Muls(tmp_local, y_local, c2, n_aligned);
            AscendC::Add(x_local, x_local, tmp_local, n_aligned);

            // 5. Apply condition (match numpy reference):
            // cond = abs(x) > 1
            // x = where(cond, sign(x), x)
            // y = where(cond, 0, y)
            AscendC::Sign(sx_local, x_local, n_aligned);
            AscendC::Abs(tmp_local, x_local, n_aligned);
            AscendC::CompareScalar<float, uint8_t>(mask_local, tmp_local, 1.0f, AscendC::CMPMODE::GT, n_aligned);
            AscendC::Select(x_local, mask_local, sx_local, x_local,
                            AscendC::SELMODE::VSEL_CMPMASK_SPR, n_aligned);
            AscendC::Duplicate(tmp_local, 0.0f, n_aligned);
            AscendC::Select(y_local, mask_local, tmp_local, y_local,
                            AscendC::SELMODE::VSEL_CMPMASK_SPR, n_aligned);

            if (n_aligned > n) {
                AscendC::Duplicate(x_local[n], 0.0f, n_aligned - n);
                AscendC::Duplicate(y_local[n], 0.0f, n_aligned - n);
            }
        }

        // Copy Out
        AscendC::DataCopy(x_gm, x_local, n);
        AscendC::DataCopy(y_gm, y_local, n);

        ub_x.FreeTensor(x_local);
        ub_y.FreeTensor(y_local);
        ub_sx.FreeTensor(sx_local);
        ub_tmp.FreeTensor(tmp_local);
        ub_J_row0.FreeTensor(J_row0);
        ub_J_row1.FreeTensor(J_row1);
        ub_mv_res.FreeTensor(mv_res_local);
        ub_work.FreeTensor(work_local);
        ub_h.FreeTensor(h_local);
        ub_reduce_out.FreeTensor(reduce_out_local);
        ub_mask.FreeTensor(mask_local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> J_gm;
    AscendC::GlobalTensor<float> x_gm, y_gm, p_gm, h_gm;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> ub_x, ub_y, ub_sx, ub_tmp, ub_J_row0, ub_J_row1,
        ub_mv_res, ub_work,
        ub_h, ub_reduce_out, ub_mask;

    float delta, xi, dt;
    int32_t niter, n, n_aligned, tile_elems;
    int32_t has_h;
};

extern "C" __global__ __aicore__ void kernel_sb_custom(GM_ADDR J, GM_ADDR x, GM_ADDR y, GM_ADDR p, GM_ADDR h,
                                                           int32_t has_h, float delta, float xi, float dt, int32_t niter, int32_t n, int32_t batch_size)
{
    int32_t batch_idx = AscendC::GetBlockIdx();
    if (batch_idx >= batch_size) return;

    // Offset inputs for the specific batch
    // J is shared
    // p is shared
    // x and y are batched
    
    // Pointers are byte-addressed in GM_ADDR usually, but here we cast to float* then add offset?
    // GM_ADDR is essentially void*. 
    // We need to be careful. __gm__ float* allows pointer arithmetic.
    
    __gm__ float* x_ptr = (__gm__ float*)x + batch_idx * n;
    __gm__ float* y_ptr = (__gm__ float*)y + batch_idx * n;

    KernelSbKernel op;
    op.Init(J, (GM_ADDR)x_ptr, (GM_ADDR)y_ptr, p, h, has_h, delta, xi, dt, niter, n);
    op.Process();
}


class KernelBsbKernel
{
public:
    __aicore__ inline KernelBsbKernel() {}
    __aicore__ inline void Init(GM_ADDR J, GM_ADDR x, GM_ADDR y, GM_ADDR p, GM_ADDR h, int32_t has_h,
                                float delta, float xi, float dt, int32_t niter, int32_t n)
    {
        this->delta = delta;
        this->xi = xi;
        this->dt = dt;
        this->niter = niter;
        this->n = n;
        // Vector operators (e.g. Mul/ReduceSum) require the length to be aligned.
        // Keep the math defined on the original `n`, and pad the tail with zeros.
        constexpr int32_t kAlign = 16;
        this->n_aligned = ((n + kAlign - 1) / kAlign) * kAlign;
        this->has_h = has_h;

        // Global Buffers
        // J is N x N
        J_gm.SetGlobalBuffer((__gm__ float *)J, n * n);
        // x and y are N (for this batch item)
        x_gm.SetGlobalBuffer((__gm__ float *)x, n);
        y_gm.SetGlobalBuffer((__gm__ float *)y, n);
        // p is niter
        p_gm.SetGlobalBuffer((__gm__ float *)p, niter);
        if (this->has_h != 0) {
            // h is N
            h_gm.SetGlobalBuffer((__gm__ float *)h, n);
        }

        // Allocate UB sized to the runtime problem size.
        // Note: UB capacity is limited; this kernel assumes `n` is small enough
        // to fit the required buffers in UB.
        uint32_t buf_size = static_cast<uint32_t>(n_aligned) * sizeof(float);
        uint32_t mask_size = static_cast<uint32_t>((n_aligned + 7) / 8) * sizeof(uint8_t);

        pipe.InitBuffer(ub_x, 1, buf_size);
        pipe.InitBuffer(ub_y, 1, buf_size);
        pipe.InitBuffer(ub_sx, 1, buf_size);
        pipe.InitBuffer(ub_tmp, 1, buf_size);
        pipe.InitBuffer(ub_mv_res, 1, buf_size);
        pipe.InitBuffer(ub_work, 1, buf_size);
        pipe.InitBuffer(ub_h, 1, buf_size);
        pipe.InitBuffer(ub_mask, 1, mask_size);
    }

    __aicore__ inline void Process()
    {
        // Load x and y
        AscendC::LocalTensor<float> x_local = ub_x.AllocTensor<float>();
        AscendC::LocalTensor<float> y_local = ub_y.AllocTensor<float>();

        AscendC::DataCopy(x_local, x_gm, n);
        AscendC::DataCopy(y_local, y_gm, n);
        event_t eventIdMte2ToVXY = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVXY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVXY);
        if (n_aligned > n) {
            AscendC::Duplicate(x_local[n], 0.0f, n_aligned - n);
            AscendC::Duplicate(y_local[n], 0.0f, n_aligned - n);
        }

        AscendC::LocalTensor<float> sx_local = ub_sx.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp_local = ub_tmp.AllocTensor<float>();
        AscendC::LocalTensor<float> J_row0 = ub_J_row0.AllocTensor<float>();
        AscendC::LocalTensor<float> J_row1 = ub_J_row1.AllocTensor<float>();
        AscendC::LocalTensor<float> mv_res_local = ub_mv_res.AllocTensor<float>();
        AscendC::LocalTensor<float> work_local = ub_work.AllocTensor<float>();
        AscendC::LocalTensor<float> h_local = ub_h.AllocTensor<float>();
        AscendC::LocalTensor<float> reduce_out_local = ub_reduce_out.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> mask_local = ub_mask.AllocTensor<uint8_t>();

        if (has_h != 0) {
            AscendC::DataCopy(h_local, h_gm, n);
            event_t eventIdMte2ToVH = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVH);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVH);
            if (n_aligned > n) {
                AscendC::Duplicate(h_local[n], 0.0f, n_aligned - n);
            }
        } else {
            AscendC::Duplicate(h_local, 0.0f, n_aligned);
        }

        for (int32_t iter = 0; iter < niter; ++iter)
        {
            // Read p[iter]
            float p_val = p_gm.GetValue(iter);

            // 1. Calculate J.dot(x)
            AscendC::Duplicate(mv_res_local, 0.0f, n_aligned);
            event_t eventIdVToS = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
            if (n <= tile_elems) {
                const int32_t chunk = n;
                event_t eventIdMte2ToV0 = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                event_t eventIdMte2ToV1 = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                AscendC::DataCopy(J_row0, J_gm[0], chunk);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV0);

                int32_t cur_buf = 0;
                for (int32_t i = 0; i < n; ++i) {
                    event_t cur_event = (cur_buf == 0) ? eventIdMte2ToV0 : eventIdMte2ToV1;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(cur_event);

                    int32_t next_buf = cur_buf ^ 1;
                    if (i + 1 < n) {
                        if (next_buf == 0) {
                            AscendC::DataCopy(J_row0, J_gm[(i + 1) * n], chunk);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV0);
                        } else {
                            AscendC::DataCopy(J_row1, J_gm[(i + 1) * n], chunk);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV1);
                        }
                    }

                    AscendC::LocalTensor<float> J_row = (cur_buf == 0) ? J_row0 : J_row1;
                    if (chunk < tile_elems) {
                        AscendC::Duplicate(J_row[chunk], 0.0f, tile_elems - chunk);
                    }
                    AscendC::Mul(tmp_local, J_row, x_local, tile_elems);
                    AscendC::ReduceSum(reduce_out_local, tmp_local, work_local, tile_elems);
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
                    mv_res_local.SetValue(i, reduce_out_local.GetValue(0));

                    cur_buf = next_buf;
                }
            } else {
                event_t eventIdMte2ToV = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                for (int32_t i = 0; i < n; ++i)
                {
                    float acc = 0.0f;
                    for (int32_t k = 0; k < n; k += tile_elems) {
                        int32_t chunk = n - k;
                        if (chunk > tile_elems) {
                            chunk = tile_elems;
                        }

                        AscendC::DataCopy(J_row0, J_gm[i * n + k], chunk);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                        if (chunk < tile_elems) {
                            AscendC::Duplicate(J_row0[chunk], 0.0f, tile_elems - chunk);
                        }

                        AscendC::Mul(tmp_local, J_row0, x_local[k], tile_elems);
                        AscendC::ReduceSum(reduce_out_local, tmp_local, work_local, tile_elems);
                        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
                        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
                        acc += reduce_out_local.GetValue(0);
                    }
                    mv_res_local.SetValue(i, acc);
                }
            }
            event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);

            // 2. Update y
            // Term 1: (-delta - p[i]) * x -> (-delta + p_val) * x
            float c1 = -delta + p_val;
            AscendC::Muls(tmp_local, x_local, c1, n_aligned);

            // Term 2: xi * mv_res
            if (has_h != 0) {
                AscendC::Add(work_local, mv_res_local, h_local, n_aligned);
                AscendC::Muls(work_local, work_local, xi, n_aligned);
            } else {
                AscendC::Muls(work_local, mv_res_local, xi, n_aligned);
            }

            // Sum terms
            AscendC::Add(tmp_local, tmp_local, work_local, n_aligned);

            // Multiply by dt
            AscendC::Muls(tmp_local, tmp_local, dt, n_aligned);

            // Update y
            AscendC::Add(y_local, y_local, tmp_local, n_aligned);

            // 3. Update x
            // x = x + dt * y * delta
            float c2 = dt * delta;
            AscendC::Muls(tmp_local, y_local, c2, n_aligned);
            AscendC::Add(x_local, x_local, tmp_local, n_aligned);

            // 4. Apply condition (match numpy reference):
            // cond = abs(x) > 1
            // x = where(cond, sign(x), x)
            // y = where(cond, 0, y)
            AscendC::Sign(sx_local, x_local, n_aligned);
            AscendC::Abs(tmp_local, x_local, n_aligned);
            AscendC::CompareScalar<float, uint8_t>(mask_local, tmp_local, 1.0f, AscendC::CMPMODE::GT, n_aligned);
            AscendC::Select(x_local, mask_local, sx_local, x_local,
                            AscendC::SELMODE::VSEL_CMPMASK_SPR, n_aligned);
            AscendC::Duplicate(tmp_local, 0.0f, n_aligned);
            AscendC::Select(y_local, mask_local, tmp_local, y_local,
                            AscendC::SELMODE::VSEL_CMPMASK_SPR, n_aligned);

            if (n_aligned > n) {
                AscendC::Duplicate(x_local[n], 0.0f, n_aligned - n);
                AscendC::Duplicate(y_local[n], 0.0f, n_aligned - n);
            }
        }

        // Copy Out
        AscendC::DataCopy(x_gm, x_local, n);
        AscendC::DataCopy(y_gm, y_local, n);

        ub_x.FreeTensor(x_local);
        ub_y.FreeTensor(y_local);
        ub_sx.FreeTensor(sx_local);
        ub_tmp.FreeTensor(tmp_local);
        ub_J_row0.FreeTensor(J_row0);
        ub_J_row1.FreeTensor(J_row1);
        ub_mv_res.FreeTensor(mv_res_local);
        ub_work.FreeTensor(work_local);
        ub_h.FreeTensor(h_local);
        ub_reduce_out.FreeTensor(reduce_out_local);
        ub_mask.FreeTensor(mask_local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> J_gm;
    AscendC::GlobalTensor<float> x_gm, y_gm, p_gm, h_gm;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> ub_x, ub_y, ub_sx, ub_tmp, ub_J_row0, ub_J_row1,
        ub_mv_res, ub_work,
        ub_h, ub_reduce_out, ub_mask;

    float delta, xi, dt;
    int32_t niter, n, n_aligned, tile_elems;
    int32_t has_h;
};

extern "C" __global__ __aicore__ void kernel_bsb_custom(GM_ADDR J, GM_ADDR x, GM_ADDR y, GM_ADDR p, GM_ADDR h,
                                                           int32_t has_h, float delta, float xi, float dt, int32_t niter, int32_t n, int32_t batch_size)
{
    int32_t batch_idx = AscendC::GetBlockIdx();
    if (batch_idx >= batch_size) return;

    // Offset inputs for the specific batch
    // J is shared
    // p is shared
    // x and y are batched

    // Pointers are byte-addressed in GM_ADDR usually, but here we cast to float* then add offset?
    // GM_ADDR is essentially void*.
    // We need to be careful. __gm__ float* allows pointer arithmetic.

    __gm__ float* x_ptr = (__gm__ float*)x + batch_idx * n;
    __gm__ float* y_ptr = (__gm__ float*)y + batch_idx * n;

    KernelBsbKernel op;
    op.Init(J, (GM_ADDR)x_ptr, (GM_ADDR)y_ptr, p, h, has_h, delta, xi, dt, niter, n);
    op.Process();
}
