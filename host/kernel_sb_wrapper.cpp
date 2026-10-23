#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclrtlaunch_kernel_sb_step.h"

namespace py = pybind11;

namespace {
void Check(int status, const char* call)
{
    if (status != 0) {
        const char* detail = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(call) + " failed (" + std::to_string(status) + ")" +
                                 (detail ? std::string(": ") + detail : ""));
    }
}
#define ACL_CHECK(call) Check((call), #call)

// One owner for all resources referenced by a captured graph, including during
// partially completed construction. No process-global device buffer cache.
struct Resources {
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    aclrtEvent start = nullptr, end = nullptr;
    aclmdlRI graph = nullptr;
    aclOpExecutor* executor = nullptr;
    bool capturing = false;
    std::vector<void*> device;
    std::vector<void*> pinned;
    std::vector<aclTensor*> tensors;

    void Init(int deviceId)
    {
        static const bool initialized = [] {
            const auto status = aclInit(nullptr);
            if (status != ACL_ERROR_REPEAT_INITIALIZE) ACL_CHECK(status);
            return true;
        }();
        (void)initialized;
        ACL_CHECK(aclrtCreateContext(&context, deviceId));
        ACL_CHECK(aclrtCreateStream(&stream));
        ACL_CHECK(aclrtCreateEvent(&start));
        ACL_CHECK(aclrtCreateEvent(&end));
    }

    void* Allocate(size_t bytes, bool host = false)
    {
        if (bytes == 0) return nullptr;
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

    aclTensor* Matrix(void* ptr, int64_t rows, int64_t cols, bool transpose = false)
    {
        const int64_t shape[] = {rows, cols};
        const int64_t strides[] = {transpose ? 1 : cols, transpose ? rows : 1};
        auto* tensor = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, ptr);
        if (!tensor) throw std::runtime_error("aclCreateTensor failed");
        tensors.push_back(tensor);
        return tensor;
    }

    ~Resources()
    {
        if (!context) return;
        aclrtSetCurrentContext(context);
        if (capturing) aclmdlRICaptureEnd(stream, &graph);
        if (stream) aclrtSynchronizeStream(stream);
        if (graph) aclmdlRIDestroy(graph);
        if (executor) aclDestroyAclOpExecutor(executor);
        for (auto* tensor : tensors) aclDestroyTensor(tensor);
        for (void* ptr : device) aclrtFree(ptr);
        for (void* ptr : pinned) aclrtFreeHost(ptr);
        if (start) aclrtDestroyEvent(start);
        if (end) aclrtDestroyEvent(end);
        if (stream) aclrtDestroyStream(stream);
        aclrtDestroyContext(context);
    }
};

using CArray = py::array_t<float, py::array::c_style>;
using FArray = py::array_t<float, py::array::f_style>;

class Plan {
public:
    Plan(CArray J, CArray h, CArray p, int32_t batch, bool discrete,
         float delta, float xi, float dt, int deviceId, int cubeMathType)
        : resources_(std::make_unique<Resources>()), batch_(batch)
    {
        if (J.ndim() != 2 || J.shape(0) != J.shape(1) || J.shape(0) <= 0 ||
            J.shape(0) > std::numeric_limits<int32_t>::max() || J.shape(0) % 64 != 0 ||
            batch <= 0 || J.shape(0) * batch > std::numeric_limits<int32_t>::max()) {
            throw std::invalid_argument("native J must be square, nonempty, padded to 64; invalid state size");
        }
        n_ = static_cast<int32_t>(J.shape(0));
        if (h.ndim() != 1 || h.shape(0) != n_ || p.ndim() != 1 || p.size() <= 0 ||
            p.size() > std::numeric_limits<int32_t>::max()) {
            throw std::invalid_argument("native h/p shape mismatch");
        }
        if ((cubeMathType != 0 && cubeMathType != 3) || !std::isfinite(delta) ||
            !std::isfinite(xi) || !std::isfinite(dt)) {
            throw std::invalid_argument("invalid math mode or integration parameters");
        }
        bytes_ = static_cast<size_t>(n_) * batch_ * sizeof(float);
        py::gil_scoped_release release;
        auto& r = *resources_;
        r.Init(deviceId);
        int64_t vectorCores = 0;
        ACL_CHECK(aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCores));
        const uint32_t blocks = static_cast<uint32_t>(std::min<int64_t>(
            vectorCores, static_cast<int64_t>(batch_) * n_ / 64));
        void* j = r.Allocate(J.nbytes());
        void* field = r.Allocate(bytes_);
        void* pressure = r.Allocate(p.nbytes());
        x_ = r.Allocate(bytes_);
        y_ = r.Allocate(bytes_);
        void* spin = discrete ? r.Allocate(bytes_) : x_;
        void* coupling = r.Allocate(bytes_);
        hostX_ = r.Allocate(bytes_, true);
        hostY_ = r.Allocate(bytes_, true);
        ACL_CHECK(aclrtMemcpy(j, J.nbytes(), J.data(), J.nbytes(), ACL_MEMCPY_HOST_TO_DEVICE));
        // Expand the immutable field once so update tiles can cross replicas.
        // Reuse staging memory; this adds no host allocation or per-run packing.
        for (int32_t replica = 0; replica < batch_; ++replica) {
            std::memcpy(static_cast<float*>(hostX_) + static_cast<int64_t>(replica) * n_,
                        h.data(), h.nbytes());
        }
        ACL_CHECK(aclrtMemcpy(field, bytes_, hostX_, bytes_, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(pressure, p.nbytes(), p.data(), p.nbytes(), ACL_MEMCPY_HOST_TO_DEVICE));

        // Transposed views: (J @ X).T == X.T @ J.T, including nonsymmetric J.
        auto* a = r.Matrix(spin, batch_, n_);
        auto* b = r.Matrix(j, n_, n_, true);
        auto* c = r.Matrix(coupling, batch_, n_);
        uint64_t workspaceBytes = 0;
        ACL_CHECK(aclnnMatmulGetWorkspaceSize(a, b, c, cubeMathType, &workspaceBytes, &r.executor));
        ACL_CHECK(aclSetAclOpExecutorRepeatable(r.executor));
        void* workspace = r.Allocate(workspaceBytes);
        auto step = [&](int32_t iteration) {
            ACL_CHECK(aclrtlaunch_kernel_sb_step(blocks, r.stream, x_, y_, coupling, spin,
                field, pressure, n_, batch_, iteration, discrete, delta, xi, dt));
        };
        auto matmul = [&] {
            ACL_CHECK(aclnnMatmul(workspace, workspaceBytes, r.executor, r.stream));
        };

        // Resolve lazy operator/kernel loading before capture. This work is
        // construction cost, never included in replay's device timing.
        ACL_CHECK(aclrtMemset(x_, bytes_, 0, bytes_));
        ACL_CHECK(aclrtMemset(y_, bytes_, 0, bytes_));
        if (discrete) step(-1);
        matmul();
        step(0);
        ACL_CHECK(aclrtSynchronizeStream(r.stream));

        ACL_CHECK(aclmdlRICaptureBegin(r.stream, ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL));
        r.capturing = true;
        if (discrete) step(-1);
        for (int32_t iteration = 0; iteration < p.size(); ++iteration) {
            matmul();
            step(iteration);
        }
        ACL_CHECK(aclmdlRICaptureEnd(r.stream, &r.graph));
        r.capturing = false;
    }

    py::tuple Run(FArray x, FArray y)
    {
        if (x.ndim() != 2 || y.ndim() != 2 || x.shape(0) != n_ || y.shape(0) != n_ ||
            x.shape(1) != batch_ || y.shape(1) != batch_) {
            throw std::invalid_argument("state must have the plan's padded (N, batch) shape");
        }
        FArray outX({n_, batch_}), outY({n_, batch_});
        {
            // Release the GIL BEFORE acquiring the plan lock: another Python
            // thread can otherwise deadlock while the running call reacquires it.
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            if (!resources_) throw std::runtime_error("Solver is closed");
            auto& r = *resources_;
            ACL_CHECK(aclrtSetCurrentContext(r.context));
            std::memcpy(hostX_, x.data(), bytes_);
            std::memcpy(hostY_, y.data(), bytes_);
            ACL_CHECK(aclrtMemcpyAsync(x_, bytes_, hostX_, bytes_, ACL_MEMCPY_HOST_TO_DEVICE, r.stream));
            ACL_CHECK(aclrtMemcpyAsync(y_, bytes_, hostY_, bytes_, ACL_MEMCPY_HOST_TO_DEVICE, r.stream));
            ACL_CHECK(aclrtRecordEvent(r.start, r.stream));
            ACL_CHECK(aclmdlRIExecuteAsync(r.graph, r.stream));
            ACL_CHECK(aclrtRecordEvent(r.end, r.stream));
            ACL_CHECK(aclrtMemcpyAsync(hostX_, bytes_, x_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST, r.stream));
            ACL_CHECK(aclrtMemcpyAsync(hostY_, bytes_, y_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST, r.stream));
            ACL_CHECK(aclrtSynchronizeStream(r.stream));
            float ms = 0;
            ACL_CHECK(aclrtEventElapsedTime(&ms, r.start, r.end));
            lastDeviceMs_ = ms;
            std::memcpy(outX.mutable_data(), hostX_, bytes_);
            std::memcpy(outY.mutable_data(), hostY_, bytes_);
        }
        return py::make_tuple(outX, outY);
    }

    void Close()
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        resources_.reset();
    }

    float LastDeviceMs() const { return lastDeviceMs_.load(); }

private:
    std::unique_ptr<Resources> resources_;
    std::mutex mutex_;
    int32_t n_, batch_;
    size_t bytes_;
    void *x_ = nullptr, *y_ = nullptr, *hostX_ = nullptr, *hostY_ = nullptr;
    std::atomic<float> lastDeviceMs_{0};
};
}  // namespace

PYBIND11_MODULE(_kernel_sb, m)
{
    m.doc() = "CANN 9.1 batched simulated bifurcation execution plan";
    py::class_<Plan>(m, "Plan")
        .def(py::init<CArray, CArray, CArray, int32_t, bool, float, float, float, int, int>(),
             py::arg("J").noconvert(), py::arg("h").noconvert(), py::arg("p").noconvert(),
             py::arg("batch"), py::arg("discrete"), py::arg("delta"), py::arg("xi"),
             py::arg("dt"), py::arg("device_id"), py::arg("cube_math_type"))
        .def("run", &Plan::Run, py::arg("x").noconvert(), py::arg("y").noconvert())
        .def("close", &Plan::Close)
        .def_property_readonly("last_device_ms", &Plan::LastDeviceMs);
}
