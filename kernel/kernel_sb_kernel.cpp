/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr int32_t kTile = 4096;

// All state uses [batch, padded_n] storage, including the read-only field.
// Partition the entire batch evenly across AIV cores, not by replica: otherwise
// 100 replicas on 48 cores leave some cores doing 50% more work than others.
// UB use is fixed (112 KiB + mask), leaving the API's reserved 8 KiB untouched.
// CompareScalar requires complete 64-element FP32 repeats, including the tail.
class StateUpdate {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR coupling, GM_ADDR spin,
                              GM_ADDR h, GM_ADDR p, int32_t n, int32_t batch,
                              int32_t iteration, int32_t discrete,
                              float delta, float xi, float dt)
    {
        GlobalTensor<float> gx, gy, gc, gs, gh, gp;
        gx.SetGlobalBuffer((__gm__ float*)x);
        gy.SetGlobalBuffer((__gm__ float*)y);
        gc.SetGlobalBuffer((__gm__ float*)coupling);
        gs.SetGlobalBuffer((__gm__ float*)spin);
        gh.SetGlobalBuffer((__gm__ float*)h);
        gp.SetGlobalBuffer((__gm__ float*)p);
        pipe.InitBuffer(values, 7 * kTile * sizeof(float));
        pipe.InitBuffer(masks, kTile / 8);
        auto lx = values.Get<float>();
        auto ly = lx[kTile];
        auto lc = lx[2 * kTile];
        auto lh = lx[3 * kTile];
        auto tmp = lx[4 * kTile];
        auto work = lx[5 * kTile];
        auto sx = lx[6 * kTile];
        auto mask = masks.Get<uint8_t>();
        const int64_t total = static_cast<int64_t>(n) * batch;
        const int64_t perCore = ((total / 64 + GetBlockNum() - 1) / GetBlockNum()) * 64;
        const int64_t start = GetBlockIdx() * perCore;
        const int64_t end = start + perCore < total ? start + perCore : total;
        const float pressure = iteration < 0 ? 0.0f : gp.GetValue(iteration);

        for (int64_t offset = start; offset < end; offset += kTile) {
            const int32_t count = end - offset < kTile ? end - offset : kTile;
            DataCopy(lx, gx[offset], count);
            if (iteration >= 0) {
                DataCopy(ly, gy[offset], count);
                DataCopy(lc, gc[offset], count);
                DataCopy(lh, gh[offset], count);
            }
            Sync<HardEvent::MTE2_V>();
            if (iteration < 0) {
                Sign(sx, lx, count);
            } else {
                // Preserve the reference's kick -> drift -> inelastic wall order.
                Muls(tmp, lx, pressure - delta, count);
                Add(work, lc, lh, count);
                PipeBarrier<PIPE_V>();
                Muls(work, work, xi, count);
                PipeBarrier<PIPE_V>();
                Add(tmp, tmp, work, count);
                PipeBarrier<PIPE_V>();
                Muls(tmp, tmp, dt, count);
                PipeBarrier<PIPE_V>();
                Add(ly, ly, tmp, count);
                PipeBarrier<PIPE_V>();
                Muls(tmp, ly, dt * delta, count);
                PipeBarrier<PIPE_V>();
                Add(lx, lx, tmp, count);
                PipeBarrier<PIPE_V>();
                Abs(tmp, lx, count);
                PipeBarrier<PIPE_V>();
                CompareScalar<float, uint8_t>(mask, tmp, 1.0f, CMPMODE::GT, count);
                Duplicate(work, 0.0f, count);
                Maxs(lx, lx, -1.0f, count);
                PipeBarrier<PIPE_V>();
                Mins(lx, lx, 1.0f, count);
                // Consume the whole memory mask, not a repeated CMPMASK register.
                Select(ly, mask, work, ly, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
                PipeBarrier<PIPE_V>();
                if (discrete) Sign(sx, lx, count);
            }
            Sync<HardEvent::V_MTE3>();
            if (iteration >= 0) {
                DataCopy(gx[offset], lx, count);
                DataCopy(gy[offset], ly, count);
            }
            if (discrete) DataCopy(gs[offset], sx, count);
            // Do not overwrite local storage until all stores have consumed it.
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

private:
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        const auto id = static_cast<event_t>(pipe.FetchEventID(event));
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> values, masks;
};
}  // namespace

extern "C" __global__ __aicore__ void kernel_sb_step(
    GM_ADDR x, GM_ADDR y, GM_ADDR coupling, GM_ADDR spin, GM_ADDR h, GM_ADDR p,
    int32_t n, int32_t batch, int32_t iteration, int32_t discrete,
    float delta, float xi, float dt)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    StateUpdate update;
    update.Run(x, y, coupling, spin, h, p, n, batch, iteration, discrete, delta, xi, dt);
}
