// Copyright 2026 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.
#include <cmath>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_matmul.h"
#include "algorithm/qaia/ascend/kernel_launch.h"
#include "core/ascend.h"

namespace py = pybind11;

namespace {
using mindquantum::algorithm::qaia::ascend::LaunchSbStep;
using mindquantum::ascend::Check;
using mindquantum::ascend::Cleanup;
using mindquantum::ascend::ContextScope;
#define ACL_CHECK(call)   Check((call), #call)
#define ACL_CLEANUP(call) Cleanup((call), #call)

// One owner for all resources referenced by a captured graph, including during
// partially completed construction. No process-global device buffer cache.
struct Resources {
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    aclmdlRI graph = nullptr;
    aclOpExecutor* executor = nullptr;
    bool capturing = false;
    std::vector<void*> device;
    std::vector<void*> pinned;
    std::vector<aclTensor*> tensors;

    void Init(int deviceId) {
        ACL_CHECK(aclrtCreateContext(&context, deviceId));
        mindquantum::ascend::SocName(MQ_ASCEND_SOC);
        ACL_CHECK(aclrtCreateStream(&stream));
    }

    void* Allocate(size_t bytes, bool host = false) {
        if (bytes == 0)
            return nullptr;
        void* ptr = nullptr;
        if (host) {
            ACL_CHECK(aclrtMallocHost(&ptr, bytes));
            pinned.push_back(ptr);
        } else {
            ACL_CHECK(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
            device.push_back(ptr);
        }
        return ptr;
    }

    aclTensor* Matrix(void* ptr, int64_t rows, int64_t cols, bool transpose = false) {
        const int64_t shape[] = {rows, cols};
        const int64_t strides[] = {transpose ? 1 : cols, transpose ? rows : 1};
        auto* tensor = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, ptr);
        if (!tensor)
            throw std::runtime_error("aclCreateTensor failed");
        tensors.push_back(tensor);
        return tensor;
    }

    ~Resources() {
        if (!context)
            return;
        const ContextScope scope(context, Cleanup);
        if (capturing)
            ACL_CLEANUP(aclmdlRICaptureEnd(stream, &graph));
        if (stream)
            ACL_CLEANUP(aclrtSynchronizeStream(stream));
        if (graph)
            ACL_CLEANUP(aclmdlRIDestroy(graph));
        if (executor)
            ACL_CLEANUP(aclDestroyAclOpExecutor(executor));
        for (auto* tensor : tensors)
            ACL_CLEANUP(aclDestroyTensor(tensor));
        for (void* ptr : device)
            ACL_CLEANUP(aclrtFree(ptr));
        for (void* ptr : pinned)
            ACL_CLEANUP(aclrtFreeHost(ptr));
        if (stream)
            ACL_CLEANUP(aclrtDestroyStream(stream));
        ACL_CLEANUP(aclrtDestroyContext(context));
    }
};

using CArray = py::array_t<float, py::array::c_style>;
using FArray = py::array_t<float, py::array::f_style>;
using StateArray = py::array_t<float, 0>;

// The same pure check serves Python's pre-allocation validation and native
// construction. Keep the device alignment and index limits in this layer only.
int32_t PaddedSize(int64_t n, int64_t batch, int64_t steps) {
    constexpr auto limit = std::numeric_limits<int32_t>::max();
    if (n <= 0 || n > limit)
        throw std::invalid_argument("J must have a positive int32 dimension");
    if (batch <= 0 || batch > limit)
        throw std::invalid_argument("batch_size must be a positive int32");
    if (steps <= 0 || steps > limit)
        throw std::invalid_argument("n_iter must be a positive int32");
    // CompareScalar consumes full 256-byte FP32 repeats on 910B.
    const auto padded = (n + 63) / 64 * 64;
    if (padded > limit / batch)
        throw std::invalid_argument("padded state exceeds int32 indexing");
    return static_cast<int32_t>(padded);
}

int32_t ValidateLayout(const CArray& J, const CArray& h, const CArray& p, int32_t batch) {
    if (J.ndim() != 2 || J.shape(0) != J.shape(1)) {
        throw std::invalid_argument("native J must be square");
    }
    if (h.ndim() != 1 || h.shape(0) != J.shape(0) || p.ndim() != 1) {
        throw std::invalid_argument("native h/p shape mismatch");
    }
    return PaddedSize(J.shape(0), batch, p.size());
}

class Plan {
 public:
    Plan(CArray J, CArray h, CArray p, int32_t batch, bool discrete, float delta, float xi, float dt, int deviceId)
        : batch_(batch) {
        paddedN_ = ValidateLayout(J, h, p, batch);
        n_ = static_cast<int32_t>(J.shape(0));
        if (!std::isfinite(delta) || !std::isfinite(xi) || !std::isfinite(dt) || delta <= 0 || dt <= 0
            || deviceId < 0) {
            throw std::invalid_argument("invalid integration parameters or device_id");
        }
        bytes_ = static_cast<size_t>(paddedN_) * batch_ * sizeof(float);
        const auto *jData = J.data(), *hData = h.data(), *pData = p.data();
        const auto jPitch = static_cast<size_t>(paddedN_) * sizeof(float);
        const auto jBytes = jPitch * paddedN_;
        const auto hBytes = h.nbytes(), pBytes = p.nbytes();
        const auto steps = static_cast<int32_t>(p.size());
        py::gil_scoped_release release;
        mindquantum::ascend::Initialize();
        const ContextScope caller;
        // On failure, release partial resources before restoring the caller's
        // context or reacquiring the GIL. Publish ownership only after capture.
        auto resources = std::make_unique<Resources>();
        auto& r = *resources;
        r.Init(deviceId);
        int64_t vectorCores = 0;
        ACL_CHECK(aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCores));
        if (vectorCores <= 0)
            throw std::runtime_error("Ascend device reports no vector cores.");
        const uint32_t blocks = static_cast<uint32_t>(
            std::min<int64_t>(vectorCores, static_cast<int64_t>(batch_) * paddedN_ / 64));
        void* j = r.Allocate(jBytes);
        void* field = r.Allocate(bytes_);
        void* pressure = r.Allocate(pBytes);
        x_ = r.Allocate(bytes_);
        y_ = r.Allocate(bytes_);
        void* spin = discrete ? r.Allocate(bytes_) : x_;
        void* coupling = r.Allocate(bytes_);
        hostX_ = r.Allocate(bytes_, true);
        hostY_ = r.Allocate(bytes_, true);
        // Upload directly into padded device rows, without a padded host matrix.
        ACL_CHECK(aclrtMemset(j, jBytes, 0, jBytes));
        ACL_CHECK(aclrtMemcpy2d(j, jPitch, jData, hBytes, hBytes, n_, ACL_MEMCPY_HOST_TO_DEVICE));
        // Expand the immutable field once so update tiles can cross replicas.
        // Reuse staging memory; this adds no host allocation or per-run packing.
        std::memset(hostX_, 0, bytes_);
        for (int32_t replica = 0; replica < batch_; ++replica) {
            std::memcpy(static_cast<float*>(hostX_) + static_cast<int64_t>(replica) * paddedN_, hData, hBytes);
        }
        ACL_CHECK(aclrtMemcpy(field, bytes_, hostX_, bytes_, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(pressure, pBytes, pData, pBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        // Transposed views: (J @ X).T == X.T @ J.T, including nonsymmetric J.
        auto* a = r.Matrix(spin, batch_, paddedN_);
        auto* b = r.Matrix(j, paddedN_, paddedN_, true);
        auto* c = r.Matrix(coupling, batch_, paddedN_);
        uint64_t workspaceBytes = 0;
        // KEEP_DTYPE: FP32 Cube arithmetic, never HF32 or automatic downcasting.
        ACL_CHECK(aclnnMatmulGetWorkspaceSize(a, b, c, 0, &workspaceBytes, &r.executor));
        ACL_CHECK(aclSetAclOpExecutorRepeatable(r.executor));
        void* workspace = r.Allocate(workspaceBytes);
        auto step = [&](int32_t iteration) {
            ACL_CHECK(LaunchSbStep(blocks, r.stream, x_, y_, coupling, spin, field, pressure, paddedN_, batch_,
                                   iteration, discrete, delta, xi, dt));
        };
        auto matmul = [&] { ACL_CHECK(aclnnMatmul(workspace, workspaceBytes, r.executor, r.stream)); };

        // Resolve lazy operator/kernel loading before capture.
        ACL_CHECK(aclrtMemset(x_, bytes_, 0, bytes_));
        ACL_CHECK(aclrtMemset(y_, bytes_, 0, bytes_));
        if (discrete)
            step(-1);
        matmul();
        step(0);
        ACL_CHECK(aclrtSynchronizeStream(r.stream));

        ACL_CHECK(aclmdlRICaptureBegin(r.stream, ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL));
        r.capturing = true;
        if (discrete)
            step(-1);
        for (int32_t iteration = 0; iteration < steps; ++iteration) {
            matmul();
            step(iteration);
        }
        ACL_CHECK(aclmdlRICaptureEnd(r.stream, &r.graph));
        r.capturing = false;
        resources_ = std::move(resources);
    }

    py::tuple Run(StateArray x, StateArray y) {
        if (x.ndim() != 2 || y.ndim() != 2 || x.shape(0) != n_ || y.shape(0) != n_ || x.shape(1) != batch_
            || y.shape(1) != batch_) {
            throw std::invalid_argument("state must have the plan's (N, batch) shape");
        }
        FArray outX({n_, batch_}), outY({n_, batch_});
        const auto xInfo = x.request(), yInfo = y.request();
        auto *outXData = outX.mutable_data(), *outYData = outY.mutable_data();
        {
            // Release the GIL BEFORE acquiring the plan lock: another Python
            // thread can otherwise deadlock while the running call reacquires it.
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            auto& r = *resources_;
            const ContextScope scope(r.context);
            mindquantum::ascend::Transaction transaction(r.stream);
            Pack(xInfo, hostX_);
            Pack(yInfo, hostY_);
            ACL_CHECK(aclrtMemcpyAsync(x_, bytes_, hostX_, bytes_, ACL_MEMCPY_HOST_TO_DEVICE, r.stream));
            ACL_CHECK(aclrtMemcpyAsync(y_, bytes_, hostY_, bytes_, ACL_MEMCPY_HOST_TO_DEVICE, r.stream));
            ACL_CHECK(aclmdlRIExecuteAsync(r.graph, r.stream));
            ACL_CHECK(aclrtMemcpyAsync(hostX_, bytes_, x_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST, r.stream));
            ACL_CHECK(aclrtMemcpyAsync(hostY_, bytes_, y_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST, r.stream));
            transaction.Finish("QAIA graph replay");
            Unpack(hostX_, outXData);
            Unpack(hostY_, outYData);
        }
        return py::make_tuple(outX, outY);
    }

 private:
    void Pack(const py::buffer_info& source, void* destination) const {
        if (n_ == paddedN_ && source.strides[0] == sizeof(float)
            && source.strides[1] == static_cast<py::ssize_t>(n_) * sizeof(float)) {
            std::memcpy(destination, source.ptr, bytes_);
            return;
        }
        for (int32_t replica = 0; replica < batch_; ++replica) {
            auto* column = static_cast<float*>(destination) + static_cast<int64_t>(replica) * paddedN_;
            const auto* input = static_cast<const char*>(source.ptr) + replica * source.strides[1];
            if (source.strides[0] == sizeof(float)) {
                std::memcpy(column, input, static_cast<size_t>(n_) * sizeof(float));
            } else {
                for (int32_t row = 0; row < n_; ++row) {
                    // Byte strides also support reversed and unaligned NumPy views.
                    std::memcpy(column + row, input + row * source.strides[0], sizeof(float));
                }
            }
            // Staging buffers contain the previous device result, not an initial state.
            std::fill(column + n_, column + paddedN_, 0.0F);
        }
    }

    void Unpack(const void* source, float* destination) const {
        if (n_ == paddedN_) {
            std::memcpy(destination, source, bytes_);
            return;
        }
        for (int32_t replica = 0; replica < batch_; ++replica) {
            std::memcpy(destination + static_cast<int64_t>(replica) * n_,
                        static_cast<const float*>(source) + static_cast<int64_t>(replica) * paddedN_,
                        static_cast<size_t>(n_) * sizeof(float));
        }
    }

    std::unique_ptr<Resources> resources_;
    std::mutex mutex_;
    int32_t n_, paddedN_, batch_;
    size_t bytes_;
    void *x_ = nullptr, *y_ = nullptr, *hostX_ = nullptr, *hostY_ = nullptr;
};
}  // namespace

PYBIND11_MODULE(_qaia_sb_ascend, m) {
    m.doc() = "CANN batched simulated bifurcation execution plan";
    py::class_<Plan>(m, "Plan")
        .def_static(
            "validate_layout", [](int64_t n, int64_t batch, int64_t steps) { PaddedSize(n, batch, steps); },
            py::arg("n"), py::arg("batch"), py::arg("steps"))
        .def(py::init<CArray, CArray, CArray, int32_t, bool, float, float, float, int>(), py::arg("J").noconvert(),
             py::arg("h").noconvert(), py::arg("p").noconvert(), py::arg("batch"), py::arg("discrete"),
             py::arg("delta"), py::arg("xi"), py::arg("dt"), py::arg("device_id"))
        .def("run", &Plan::Run, py::arg("x").noconvert(), py::arg("y").noconvert());
}
