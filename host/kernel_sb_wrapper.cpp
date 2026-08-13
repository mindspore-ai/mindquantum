#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <mutex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include <cstdint>
#include "acl/acl.h"
#include "aclrtlaunch_kernel_sb_custom.h"
#include "aclrtlaunch_kernel_bsb_custom.h"

namespace py = pybind11;

#define CHECK_ACL(x) \
    do { \
        aclError __ret = x; \
        if (__ret != ACL_ERROR_NONE) { \
            throw std::runtime_error("ACL call failed: " + std::to_string(__ret)); \
        } \
    } while (0)

namespace {

struct AclContext {
    std::mutex mu;
    bool acl_initialized{false};
    int device_id{-1};
    aclrtStream stream{nullptr};

    void* d_J{nullptr};
    void* d_x{nullptr};
    void* d_y{nullptr};
    void* d_p{nullptr};
    void* d_h{nullptr};

    size_t cap_J{0};
    size_t cap_x{0};
    size_t cap_y{0};
    size_t cap_p{0};
    size_t cap_h{0};

    const void* last_J_host_ptr{nullptr};
    size_t last_J_bytes{0};
    const void* last_p_host_ptr{nullptr};
    size_t last_p_bytes{0};
    const void* last_h_host_ptr{nullptr};
    size_t last_h_bytes{0};

    void EnsureInitialized(int requested_device_id) {
        if (!acl_initialized) {
            aclError ret = aclInit(nullptr);
            if (ret != ACL_ERROR_NONE && ret != ACL_ERROR_REPEAT_INITIALIZE) {
                throw std::runtime_error("aclInit failed: " + std::to_string(ret));
            }
            acl_initialized = true;
        }

        if (device_id != requested_device_id) {
            CHECK_ACL(aclrtSetDevice(requested_device_id));
            device_id = requested_device_id;
        }

        if (stream == nullptr) {
            CHECK_ACL(aclrtCreateStream(&stream));
        }
    }

    static void EnsureBuffer(void** buf, size_t* cap, size_t need_bytes) {
        if (need_bytes == 0) return;
        if (*buf != nullptr && *cap >= need_bytes) return;
        if (*buf != nullptr) {
            CHECK_ACL(aclrtFree(*buf));
            *buf = nullptr;
            *cap = 0;
        }
        CHECK_ACL(aclrtMalloc(buf, need_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        *cap = need_bytes;
    }
};

AclContext& GetAclContext() {
    static AclContext ctx;
    return ctx;
}

py::array EnsureDenseFloat32C(py::object obj) {
    py::module_ np = py::module_::import("numpy");
    py::object dense = obj;
    if (!py::isinstance<py::array>(dense) && py::hasattr(dense, "toarray")) {
        dense = dense.attr("toarray")();
    }
    py::object arr = np.attr("ascontiguousarray")(dense, py::arg("dtype") = np.attr("float32"));
    return arr.cast<py::array>();
}

py::array EnsureFloat32Fortran2D(py::object obj, int32_t N) {
    py::module_ np = py::module_::import("numpy");
    py::object arr = obj;
    if (!py::isinstance<py::array>(arr) && py::hasattr(arr, "toarray")) {
        arr = arr.attr("toarray")();
    }
    py::array a = np.attr("asarray")(arr, py::arg("dtype") = np.attr("float32")).cast<py::array>();
    if (a.ndim() == 1) {
        if (a.shape(0) != N) throw std::runtime_error("x/y shape mismatch");
        a = a.attr("reshape")(py::make_tuple(N, 1)).cast<py::array>();
    }
    if (a.ndim() != 2) throw std::runtime_error("x/y must be 1D or 2D");
    if (a.shape(0) != N) throw std::runtime_error("x/y shape mismatch (N)");
    return np.attr("asfortranarray")(a).cast<py::array>();
}

py::array EnsureFloat32CVector(py::object obj, int32_t N) {
    py::module_ np = py::module_::import("numpy");
    py::object arr = obj;
    if (!py::isinstance<py::array>(arr) && py::hasattr(arr, "toarray")) {
        arr = arr.attr("toarray")();
    }
    py::array a = np.attr("asarray")(arr, py::arg("dtype") = np.attr("float32")).cast<py::array>();
    if (a.ndim() == 2 && a.shape(1) == 1) {
        a = a.attr("reshape")(py::make_tuple(a.shape(0))).cast<py::array>();
    }
    if (a.ndim() != 1) throw std::runtime_error("h must be 1D (or Nx1)");
    if (a.shape(0) != N) throw std::runtime_error("h shape mismatch");
    return np.attr("ascontiguousarray")(a).cast<py::array>();
}

py::array MakeLinspaceFloat32(int32_t n) {
    py::module_ np = py::module_::import("numpy");
    if (n <= 0) {
        return py::array_t<float, py::array::c_style>(0);
    }
    py::object arr = np.attr("linspace")(0.0, 1.0, n);
    return np.attr("ascontiguousarray")(arr, py::arg("dtype") = np.attr("float32")).cast<py::array>();
}

py::array MakeRandomCenteredFloat32Fortran(int32_t N, int32_t batch_size) {
    py::array_t<float> arr(
        py::array::ShapeContainer{N, batch_size},
        py::array::StridesContainer{static_cast<py::ssize_t>(sizeof(float)),
                                    static_cast<py::ssize_t>(sizeof(float)) * N});
    float* data = static_cast<float*>(arr.mutable_data());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
    for (int32_t b = 0; b < batch_size; ++b) {
        float* col = data + static_cast<size_t>(b) * static_cast<size_t>(N);
        for (int32_t i = 0; i < N; ++i) {
            col[i] = dist(gen);
        }
    }
    return arr;
}

bool TryGetLastSbInit(int32_t N, int32_t batch_size, py::array* x_out, py::array* y_out) {
    try {
        py::module_ sb = py::module_::import("SB");
        if (!py::hasattr(sb, "_LAST_SB_INIT")) return false;
        py::object init_obj = sb.attr("_LAST_SB_INIT");
        if (init_obj.is_none()) return false;
        py::tuple init_tup = init_obj.cast<py::tuple>();
        if (init_tup.size() != 2) return false;
        py::array x = EnsureFloat32Fortran2D(init_tup[0], N);
        py::array y = EnsureFloat32Fortran2D(init_tup[1], N);
        if (x.shape(1) != batch_size || y.shape(1) != batch_size) return false;
        *x_out = x;
        *y_out = y;
        return true;
    } catch (...) {
        return false;
    }
}

float ComputeDefaultXiFromDenseJ(const py::buffer_info& J_buf) {
    if (J_buf.ndim != 2 || J_buf.shape[0] != J_buf.shape[1]) {
        throw std::runtime_error("J must be square 2D");
    }
    int32_t N = static_cast<int32_t>(J_buf.shape[0]);
    if (N <= 1) return 0.0f;
    const float* data = static_cast<const float*>(J_buf.ptr);
    double sumsq = 0.0;
    size_t nn = static_cast<size_t>(N) * static_cast<size_t>(N);
    for (size_t i = 0; i < nn; ++i) {
        double v = static_cast<double>(data[i]);
        sumsq += v * v;
    }
    if (sumsq <= 0.0) {
        throw std::runtime_error("Cannot compute default xi: sum(J^2) <= 0");
    }
    double num = 0.5 * std::sqrt(static_cast<double>(N - 1));
    return static_cast<float>(num / std::sqrt(sumsq));
}

}  // namespace

void solve(
    py::array_t<float, py::array::c_style> J_np,
    py::array_t<float, py::array::f_style> x_np,
    py::array_t<float, py::array::f_style> y_np,
    float delta, float xi, float dt,
    py::array_t<float, py::array::c_style> p_np, // Changed from float to array
    int32_t niter,
    int device_id = 0,
    py::object h_obj = py::none()
) {
    py::buffer_info J_buf = J_np.request();
    py::buffer_info x_buf = x_np.request();
    py::buffer_info y_buf = y_np.request();
    py::buffer_info p_buf = p_np.request();
    bool has_h = !h_obj.is_none();
    py::buffer_info h_buf;
    py::array_t<float, py::array::c_style | py::array::forcecast> h_np;

    if (J_buf.ndim != 2) throw std::runtime_error("J must be 2D");
    
    // x and y can be 1D (N) or 2D (N, batch_size)
    // But if batch_size is supported, they should be treated accordingly.
    // The user says x and y change to (N x batch_Size).
    // This likely means shape is (N, batch_size). Or maybe (batch_size, N)?
    // "x and y are updated, changing from (N) to (N x batch_Size)"
    // Usually means x has N rows, B columns.
    // Let's support both or assume (N, B).
    
    int32_t N = J_buf.shape[0];
    int32_t batch_size = 1;
    
    if (x_buf.ndim == 2) {
        if (x_buf.shape[0] != N) throw std::runtime_error("x shape mismatch (N)");
        batch_size = x_buf.shape[1];
    } else if (x_buf.ndim == 1) {
        if (x_buf.shape[0] != N) throw std::runtime_error("x shape mismatch");
        batch_size = 1;
    } else {
         throw std::runtime_error("x must be 1D or 2D");
    }

    // Check y
    if (y_buf.ndim != x_buf.ndim) throw std::runtime_error("y must have same dims as x");
    if (y_buf.shape[0] != N) throw std::runtime_error("y shape mismatch");
    if (x_buf.ndim == 2 && y_buf.shape[1] != batch_size) throw std::runtime_error("y batch size mismatch");
    
    // Check p
    if (p_buf.ndim != 1) throw std::runtime_error("p must be 1D");
    if (p_buf.shape[0] != niter) throw std::runtime_error("p size must match niter");

    if (has_h) {
        py::array h_arr = EnsureFloat32CVector(h_obj, N);
        h_np = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(h_arr);
        if (!h_np) throw std::runtime_error("Failed to convert h to float32 contiguous array");
        h_buf = h_np.request();
    }

    auto& ctx = GetAclContext();
    std::lock_guard<std::mutex> lock(ctx.mu);
    ctx.EnsureInitialized(device_id);

    size_t J_size = static_cast<size_t>(J_buf.size) * static_cast<size_t>(J_buf.itemsize);
    size_t vec_size = N * batch_size * sizeof(float);
    size_t p_size = niter * sizeof(float);
    size_t h_size = N * sizeof(float);

    AclContext::EnsureBuffer(&ctx.d_J, &ctx.cap_J, J_size);
    AclContext::EnsureBuffer(&ctx.d_x, &ctx.cap_x, vec_size);
    AclContext::EnsureBuffer(&ctx.d_y, &ctx.cap_y, vec_size);
    AclContext::EnsureBuffer(&ctx.d_p, &ctx.cap_p, p_size);
    AclContext::EnsureBuffer(&ctx.d_h, &ctx.cap_h, h_size);

    CHECK_ACL(aclrtMemcpyAsync(ctx.d_J, ctx.cap_J, J_buf.ptr, J_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    CHECK_ACL(aclrtMemcpyAsync(ctx.d_x, ctx.cap_x, x_buf.ptr, vec_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    CHECK_ACL(aclrtMemcpyAsync(ctx.d_y, ctx.cap_y, y_buf.ptr, vec_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    CHECK_ACL(aclrtMemcpyAsync(ctx.d_p, ctx.cap_p, p_buf.ptr, p_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    if (has_h) {
        CHECK_ACL(aclrtMemcpyAsync(ctx.d_h, ctx.cap_h, h_buf.ptr, h_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    }

    // Launch
    // Use batch_size as blockDim
    CHECK_ACL(
        aclrtlaunch_kernel_sb_custom(
            batch_size, ctx.stream, ctx.d_J, ctx.d_x, ctx.d_y, ctx.d_p, ctx.d_h, static_cast<int32_t>(has_h),
            delta, xi, dt, niter, N, batch_size));

    // Copy back
    CHECK_ACL(aclrtMemcpyAsync(x_buf.ptr, vec_size, ctx.d_x, vec_size, ACL_MEMCPY_DEVICE_TO_HOST, ctx.stream));
    CHECK_ACL(aclrtMemcpyAsync(y_buf.ptr, vec_size, ctx.d_y, vec_size, ACL_MEMCPY_DEVICE_TO_HOST, ctx.stream));
    CHECK_ACL(aclrtSynchronizeStream(ctx.stream));
}

void solve_bsb(
    py::array_t<float, py::array::c_style> J_np,
    py::array_t<float, py::array::f_style> x_np,
    py::array_t<float, py::array::f_style> y_np,
    float delta, float xi, float dt,
    py::array_t<float, py::array::c_style> p_np, // Changed from float to array
    int32_t niter,
    int device_id = 0,
    py::object h_obj = py::none()
) {
    py::buffer_info J_buf = J_np.request();
    py::buffer_info x_buf = x_np.request();
    py::buffer_info y_buf = y_np.request();
    py::buffer_info p_buf = p_np.request();
    bool has_h = !h_obj.is_none();
    py::buffer_info h_buf;
    py::array_t<float, py::array::c_style | py::array::forcecast> h_np;

    if (J_buf.ndim != 2) throw std::runtime_error("J must be 2D");

    int32_t N = J_buf.shape[0];
    int32_t batch_size = 1;

    if (x_buf.ndim == 2) {
        if (x_buf.shape[0] != N) throw std::runtime_error("x shape mismatch (N)");
        batch_size = x_buf.shape[1];
    } else if (x_buf.ndim == 1) {
        if (x_buf.shape[0] != N) throw std::runtime_error("x shape mismatch");
        batch_size = 1;
    } else {
         throw std::runtime_error("x must be 1D or 2D");
    }

    // Check y
    if (y_buf.ndim != x_buf.ndim) throw std::runtime_error("y must have same dims as x");
    if (y_buf.shape[0] != N) throw std::runtime_error("y shape mismatch");
    if (x_buf.ndim == 2 && y_buf.shape[1] != batch_size) throw std::runtime_error("y batch size mismatch");

    // Check p
    if (p_buf.ndim != 1) throw std::runtime_error("p must be 1D");
    if (p_buf.shape[0] != niter) throw std::runtime_error("p size must match niter");

    if (has_h) {
        py::array h_arr = EnsureFloat32CVector(h_obj, N);
        h_np = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(h_arr);
        if (!h_np) throw std::runtime_error("Failed to convert h to float32 contiguous array");
        h_buf = h_np.request();
    }

    auto& ctx = GetAclContext();
    std::lock_guard<std::mutex> lock(ctx.mu);
    ctx.EnsureInitialized(device_id);

    size_t J_size = static_cast<size_t>(J_buf.size) * static_cast<size_t>(J_buf.itemsize);
    size_t vec_size = N * batch_size * sizeof(float);
    size_t p_size = niter * sizeof(float);
    size_t h_size = N * sizeof(float);

    AclContext::EnsureBuffer(&ctx.d_J, &ctx.cap_J, J_size);
    AclContext::EnsureBuffer(&ctx.d_x, &ctx.cap_x, vec_size);
    AclContext::EnsureBuffer(&ctx.d_y, &ctx.cap_y, vec_size);
    AclContext::EnsureBuffer(&ctx.d_p, &ctx.cap_p, p_size);
    AclContext::EnsureBuffer(&ctx.d_h, &ctx.cap_h, h_size);

    CHECK_ACL(aclrtMemcpyAsync(ctx.d_J, ctx.cap_J, J_buf.ptr, J_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    CHECK_ACL(aclrtMemcpyAsync(ctx.d_x, ctx.cap_x, x_buf.ptr, vec_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    CHECK_ACL(aclrtMemcpyAsync(ctx.d_y, ctx.cap_y, y_buf.ptr, vec_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    CHECK_ACL(aclrtMemcpyAsync(ctx.d_p, ctx.cap_p, p_buf.ptr, p_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    if (has_h) {
        CHECK_ACL(aclrtMemcpyAsync(ctx.d_h, ctx.cap_h, h_buf.ptr, h_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    }

    // Launch
    // Use batch_size as blockDim
    CHECK_ACL(
        aclrtlaunch_kernel_bsb_custom(
            batch_size, ctx.stream, ctx.d_J, ctx.d_x, ctx.d_y, ctx.d_p, ctx.d_h, static_cast<int32_t>(has_h),
            delta, xi, dt, niter, N, batch_size));

    // Copy back
    CHECK_ACL(aclrtMemcpyAsync(x_buf.ptr, vec_size, ctx.d_x, vec_size, ACL_MEMCPY_DEVICE_TO_HOST, ctx.stream));
    CHECK_ACL(aclrtMemcpyAsync(y_buf.ptr, vec_size, ctx.d_y, vec_size, ACL_MEMCPY_DEVICE_TO_HOST, ctx.stream));
    CHECK_ACL(aclrtSynchronizeStream(ctx.stream));
}

void solve_cached(
    py::array_t<float, py::array::c_style> J_np,
    py::array_t<float, py::array::f_style> x_np,
    py::array_t<float, py::array::f_style> y_np,
    float delta, float xi, float dt,
    py::array_t<float, py::array::c_style> p_np,
    int32_t niter,
    bool force_copy = false,
    int device_id = 0,
    py::object h_obj = py::none()
) {
    py::buffer_info J_buf = J_np.request();
    py::buffer_info x_buf = x_np.request();
    py::buffer_info y_buf = y_np.request();
    py::buffer_info p_buf = p_np.request();
    bool has_h = !h_obj.is_none();
    py::buffer_info h_buf;
    py::array_t<float, py::array::c_style | py::array::forcecast> h_np;

    if (J_buf.ndim != 2) throw std::runtime_error("J must be 2D");

    int32_t N = J_buf.shape[0];
    int32_t batch_size = 1;

    if (x_buf.ndim == 2) {
        if (x_buf.shape[0] != N) throw std::runtime_error("x shape mismatch (N)");
        batch_size = x_buf.shape[1];
    } else if (x_buf.ndim == 1) {
        if (x_buf.shape[0] != N) throw std::runtime_error("x shape mismatch");
        batch_size = 1;
    } else {
        throw std::runtime_error("x must be 1D or 2D");
    }

    if (y_buf.ndim != x_buf.ndim) throw std::runtime_error("y must have same dims as x");
    if (y_buf.shape[0] != N) throw std::runtime_error("y shape mismatch");
    if (x_buf.ndim == 2 && y_buf.shape[1] != batch_size) throw std::runtime_error("y batch size mismatch");

    if (p_buf.ndim != 1) throw std::runtime_error("p must be 1D");
    if (p_buf.shape[0] != niter) throw std::runtime_error("p size must match niter");

    if (has_h) {
        py::array h_arr = EnsureFloat32CVector(h_obj, N);
        h_np = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(h_arr);
        if (!h_np) throw std::runtime_error("Failed to convert h to float32 contiguous array");
        h_buf = h_np.request();
    }

    auto& ctx = GetAclContext();
    std::lock_guard<std::mutex> lock(ctx.mu);
    ctx.EnsureInitialized(device_id);

    size_t J_size = static_cast<size_t>(J_buf.size) * static_cast<size_t>(J_buf.itemsize);
    size_t vec_size = N * batch_size * sizeof(float);
    size_t p_size = niter * sizeof(float);
    size_t h_size = N * sizeof(float);

    AclContext::EnsureBuffer(&ctx.d_J, &ctx.cap_J, J_size);
    AclContext::EnsureBuffer(&ctx.d_x, &ctx.cap_x, vec_size);
    AclContext::EnsureBuffer(&ctx.d_y, &ctx.cap_y, vec_size);
    AclContext::EnsureBuffer(&ctx.d_p, &ctx.cap_p, p_size);
    AclContext::EnsureBuffer(&ctx.d_h, &ctx.cap_h, h_size);

    bool need_copy_J = true;
    if (!force_copy && ctx.last_J_host_ptr == J_buf.ptr && ctx.last_J_bytes == J_size) {
        need_copy_J = false;
    } else {
        ctx.last_J_host_ptr = J_buf.ptr;
        ctx.last_J_bytes = J_size;
    }

    bool need_copy_p = true;
    if (!force_copy && ctx.last_p_host_ptr == p_buf.ptr && ctx.last_p_bytes == p_size) {
        need_copy_p = false;
    } else {
        ctx.last_p_host_ptr = p_buf.ptr;
        ctx.last_p_bytes = p_size;
    }

    bool need_copy_h = has_h;
    if (has_h) {
        if (!force_copy && ctx.last_h_host_ptr == h_buf.ptr && ctx.last_h_bytes == h_size) {
            need_copy_h = false;
        } else {
            ctx.last_h_host_ptr = h_buf.ptr;
            ctx.last_h_bytes = h_size;
        }
    }

    if (need_copy_J) {
        CHECK_ACL(aclrtMemcpyAsync(ctx.d_J, ctx.cap_J, J_buf.ptr, J_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    }
    CHECK_ACL(aclrtMemcpyAsync(ctx.d_x, ctx.cap_x, x_buf.ptr, vec_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    CHECK_ACL(aclrtMemcpyAsync(ctx.d_y, ctx.cap_y, y_buf.ptr, vec_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    if (need_copy_p) {
        CHECK_ACL(aclrtMemcpyAsync(ctx.d_p, ctx.cap_p, p_buf.ptr, p_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    }
    if (need_copy_h) {
        CHECK_ACL(aclrtMemcpyAsync(ctx.d_h, ctx.cap_h, h_buf.ptr, h_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    }

    CHECK_ACL(
        aclrtlaunch_kernel_sb_custom(
            batch_size, ctx.stream, ctx.d_J, ctx.d_x, ctx.d_y, ctx.d_p, ctx.d_h, static_cast<int32_t>(has_h),
            delta, xi, dt, niter, N, batch_size));

    CHECK_ACL(aclrtMemcpyAsync(x_buf.ptr, vec_size, ctx.d_x, vec_size, ACL_MEMCPY_DEVICE_TO_HOST, ctx.stream));
    CHECK_ACL(aclrtMemcpyAsync(y_buf.ptr, vec_size, ctx.d_y, vec_size, ACL_MEMCPY_DEVICE_TO_HOST, ctx.stream));
    CHECK_ACL(aclrtSynchronizeStream(ctx.stream));
}

void solve_cached_bsb(
    py::array_t<float, py::array::c_style> J_np,
    py::array_t<float, py::array::f_style> x_np,
    py::array_t<float, py::array::f_style> y_np,
    float delta, float xi, float dt,
    py::array_t<float, py::array::c_style> p_np,
    int32_t niter,
    bool force_copy = false,
    int device_id = 0,
    py::object h_obj = py::none()
) {
    py::buffer_info J_buf = J_np.request();
    py::buffer_info x_buf = x_np.request();
    py::buffer_info y_buf = y_np.request();
    py::buffer_info p_buf = p_np.request();
    bool has_h = !h_obj.is_none();
    py::buffer_info h_buf;
    py::array_t<float, py::array::c_style | py::array::forcecast> h_np;

    if (J_buf.ndim != 2) throw std::runtime_error("J must be 2D");

    int32_t N = J_buf.shape[0];
    int32_t batch_size = 1;

    if (x_buf.ndim == 2) {
        if (x_buf.shape[0] != N) throw std::runtime_error("x shape mismatch (N)");
        batch_size = x_buf.shape[1];
    } else if (x_buf.ndim == 1) {
        if (x_buf.shape[0] != N) throw std::runtime_error("x shape mismatch");
        batch_size = 1;
    } else {
        throw std::runtime_error("x must be 1D or 2D");
    }

    if (y_buf.ndim != x_buf.ndim) throw std::runtime_error("y must have same dims as x");
    if (y_buf.shape[0] != N) throw std::runtime_error("y shape mismatch");
    if (x_buf.ndim == 2 && y_buf.shape[1] != batch_size) throw std::runtime_error("y batch size mismatch");

    if (p_buf.ndim != 1) throw std::runtime_error("p must be 1D");
    if (p_buf.shape[0] != niter) throw std::runtime_error("p size must match niter");

    if (has_h) {
        py::array h_arr = EnsureFloat32CVector(h_obj, N);
        h_np = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(h_arr);
        if (!h_np) throw std::runtime_error("Failed to convert h to float32 contiguous array");
        h_buf = h_np.request();
    }

    auto& ctx = GetAclContext();
    std::lock_guard<std::mutex> lock(ctx.mu);
    ctx.EnsureInitialized(device_id);

    size_t J_size = static_cast<size_t>(J_buf.size) * static_cast<size_t>(J_buf.itemsize);
    size_t vec_size = N * batch_size * sizeof(float);
    size_t p_size = niter * sizeof(float);
    size_t h_size = N * sizeof(float);

    AclContext::EnsureBuffer(&ctx.d_J, &ctx.cap_J, J_size);
    AclContext::EnsureBuffer(&ctx.d_x, &ctx.cap_x, vec_size);
    AclContext::EnsureBuffer(&ctx.d_y, &ctx.cap_y, vec_size);
    AclContext::EnsureBuffer(&ctx.d_p, &ctx.cap_p, p_size);
    AclContext::EnsureBuffer(&ctx.d_h, &ctx.cap_h, h_size);

    bool need_copy_J = true;
    if (!force_copy && ctx.last_J_host_ptr == J_buf.ptr && ctx.last_J_bytes == J_size) {
        need_copy_J = false;
    } else {
        ctx.last_J_host_ptr = J_buf.ptr;
        ctx.last_J_bytes = J_size;
    }

    bool need_copy_p = true;
    if (!force_copy && ctx.last_p_host_ptr == p_buf.ptr && ctx.last_p_bytes == p_size) {
        need_copy_p = false;
    } else {
        ctx.last_p_host_ptr = p_buf.ptr;
        ctx.last_p_bytes = p_size;
    }

    bool need_copy_h = has_h;
    if (has_h) {
        if (!force_copy && ctx.last_h_host_ptr == h_buf.ptr && ctx.last_h_bytes == h_size) {
            need_copy_h = false;
        } else {
            ctx.last_h_host_ptr = h_buf.ptr;
            ctx.last_h_bytes = h_size;
        }
    }

    if (need_copy_J) {
        CHECK_ACL(aclrtMemcpyAsync(ctx.d_J, ctx.cap_J, J_buf.ptr, J_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    }
    CHECK_ACL(aclrtMemcpyAsync(ctx.d_x, ctx.cap_x, x_buf.ptr, vec_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    CHECK_ACL(aclrtMemcpyAsync(ctx.d_y, ctx.cap_y, y_buf.ptr, vec_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    if (need_copy_p) {
        CHECK_ACL(aclrtMemcpyAsync(ctx.d_p, ctx.cap_p, p_buf.ptr, p_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    }
    if (need_copy_h) {
        CHECK_ACL(aclrtMemcpyAsync(ctx.d_h, ctx.cap_h, h_buf.ptr, h_size, ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
    }

    CHECK_ACL(
        aclrtlaunch_kernel_bsb_custom(
            batch_size, ctx.stream, ctx.d_J, ctx.d_x, ctx.d_y, ctx.d_p, ctx.d_h, static_cast<int32_t>(has_h),
            delta, xi, dt, niter, N, batch_size));

    CHECK_ACL(aclrtMemcpyAsync(x_buf.ptr, vec_size, ctx.d_x, vec_size, ACL_MEMCPY_DEVICE_TO_HOST, ctx.stream));
    CHECK_ACL(aclrtMemcpyAsync(y_buf.ptr, vec_size, ctx.d_y, vec_size, ACL_MEMCPY_DEVICE_TO_HOST, ctx.stream));
    CHECK_ACL(aclrtSynchronizeStream(ctx.stream));
}

py::tuple kernel_dsb(
    py::object J_obj,
    py::object h_obj = py::none(),
    py::object x_obj = py::none(),
    py::object y_obj = py::none(),
    float delta = 1.0f,
    py::object xi_obj = py::none(),
    float dt = 1.0f,
    py::object p_obj = py::none(),
    int32_t n_iter = 1000,
    int32_t batch_size = 1,
    int device_id = 0
) {
    py::array J_arr = EnsureDenseFloat32C(J_obj);
    py::buffer_info J_buf = J_arr.request();
    if (J_buf.ndim != 2 || J_buf.shape[0] != J_buf.shape[1]) throw std::runtime_error("J must be square 2D");
    int32_t N = static_cast<int32_t>(J_buf.shape[0]);
    constexpr int32_t kAlign = 16;
    int32_t N_aligned = ((N + kAlign - 1) / kAlign) * kAlign;

    int32_t inferred_batch = batch_size;
    if (!x_obj.is_none()) {
        py::array x_tmp = EnsureFloat32Fortran2D(x_obj, N);
        inferred_batch = static_cast<int32_t>(x_tmp.shape(1));
    } else if (!y_obj.is_none()) {
        py::array y_tmp = EnsureFloat32Fortran2D(y_obj, N);
        inferred_batch = static_cast<int32_t>(y_tmp.shape(1));
    }

    py::array x_arr;
    py::array y_arr;
    const bool has_x = !x_obj.is_none();
    const bool has_y = !y_obj.is_none();
    if (!has_x && !has_y) {
        if (!TryGetLastSbInit(N, inferred_batch, &x_arr, &y_arr)) {
            x_arr = MakeRandomCenteredFloat32Fortran(N, inferred_batch);
            y_arr = MakeRandomCenteredFloat32Fortran(N, inferred_batch);
        }
    } else {
        x_arr = has_x ? EnsureFloat32Fortran2D(x_obj, N) : MakeRandomCenteredFloat32Fortran(N, inferred_batch);
        y_arr = has_y ? EnsureFloat32Fortran2D(y_obj, N) : MakeRandomCenteredFloat32Fortran(N, inferred_batch);
    }

    py::array p_arr;
    if (p_obj.is_none()) {
        p_arr = MakeLinspaceFloat32(n_iter);
    } else {
        py::module_ np = py::module_::import("numpy");
        py::array p_tmp = np.attr("asarray")(p_obj, py::arg("dtype") = np.attr("float32")).cast<py::array>();
        if (p_tmp.ndim() != 1) throw std::runtime_error("p must be 1D");
        if (p_tmp.shape(0) != n_iter) {
            throw std::runtime_error("p size must match n_iter");
        }
        p_arr = np.attr("ascontiguousarray")(p_tmp).cast<py::array>();
    }

    float xi = xi_obj.is_none() ? ComputeDefaultXiFromDenseJ(J_buf) : xi_obj.cast<float>();

    py::array J_use = J_arr;
    py::array x_use = x_arr;
    py::array y_use = y_arr;
    py::object h_use = h_obj;

    if (N_aligned != N) {
        py::array_t<float, py::array::c_style> J_pad({N_aligned, N_aligned});
        auto* J_pad_data = static_cast<float*>(J_pad.mutable_data());
        std::fill(J_pad_data, J_pad_data + static_cast<size_t>(N_aligned) * static_cast<size_t>(N_aligned), 0.0f);
        const float* J_data = static_cast<const float*>(J_buf.ptr);
        for (int32_t i = 0; i < N; ++i) {
            std::memcpy(J_pad_data + static_cast<size_t>(i) * static_cast<size_t>(N_aligned),
                        J_data + static_cast<size_t>(i) * static_cast<size_t>(N),
                        static_cast<size_t>(N) * sizeof(float));
        }

        py::array x_pad = MakeRandomCenteredFloat32Fortran(N_aligned, inferred_batch);
        py::array y_pad = MakeRandomCenteredFloat32Fortran(N_aligned, inferred_batch);
        std::fill(static_cast<float*>(x_pad.mutable_data()),
                  static_cast<float*>(x_pad.mutable_data()) + static_cast<size_t>(N_aligned) * static_cast<size_t>(inferred_batch),
                  0.0f);
        std::fill(static_cast<float*>(y_pad.mutable_data()),
                  static_cast<float*>(y_pad.mutable_data()) + static_cast<size_t>(N_aligned) * static_cast<size_t>(inferred_batch),
                  0.0f);

        const float* x_data = static_cast<const float*>(x_arr.request().ptr);
        const float* y_data = static_cast<const float*>(y_arr.request().ptr);
        float* x_pad_data = static_cast<float*>(x_pad.mutable_data());
        float* y_pad_data = static_cast<float*>(y_pad.mutable_data());
        for (int32_t b = 0; b < inferred_batch; ++b) {
            std::memcpy(x_pad_data + static_cast<size_t>(b) * static_cast<size_t>(N_aligned),
                        x_data + static_cast<size_t>(b) * static_cast<size_t>(N),
                        static_cast<size_t>(N) * sizeof(float));
            std::memcpy(y_pad_data + static_cast<size_t>(b) * static_cast<size_t>(N_aligned),
                        y_data + static_cast<size_t>(b) * static_cast<size_t>(N),
                        static_cast<size_t>(N) * sizeof(float));
        }

        if (!h_obj.is_none()) {
            py::array h_vec = EnsureFloat32CVector(h_obj, N);
            py::array_t<float, py::array::c_style> h_pad({N_aligned});
            auto* h_pad_data = static_cast<float*>(h_pad.mutable_data());
            std::fill(h_pad_data, h_pad_data + static_cast<size_t>(N_aligned), 0.0f);
            const float* h_data = static_cast<const float*>(h_vec.request().ptr);
            std::memcpy(h_pad_data, h_data, static_cast<size_t>(N) * sizeof(float));
            h_use = h_pad;
        }

        J_use = J_pad;
        x_use = x_pad;
        y_use = y_pad;
    }

    solve(
        J_use.cast<py::array_t<float, py::array::c_style>>(),
        x_use.cast<py::array_t<float, py::array::f_style>>(),
        y_use.cast<py::array_t<float, py::array::f_style>>(),
        delta,
        xi,
        dt,
        p_arr.cast<py::array_t<float, py::array::c_style>>(),
        n_iter,
        device_id,
        h_use);

    if (N_aligned != N) {
        py::module_ np = py::module_::import("numpy");
        py::array x_trim = np.attr("asfortranarray")(x_use.attr("__getitem__")(py::make_tuple(py::slice(0, N, 1), py::slice(0, inferred_batch, 1)))).cast<py::array>();
        py::array y_trim = np.attr("asfortranarray")(y_use.attr("__getitem__")(py::make_tuple(py::slice(0, N, 1), py::slice(0, inferred_batch, 1)))).cast<py::array>();
        return py::make_tuple(x_trim, y_trim);
    }

    return py::make_tuple(x_use, y_use);
}

py::tuple kernel_bsb(
    py::object J_obj,
    py::object h_obj = py::none(),
    py::object x_obj = py::none(),
    py::object y_obj = py::none(),
    float delta = 1.0f,
    py::object xi_obj = py::none(),
    float dt = 1.0f,
    py::object p_obj = py::none(),
    int32_t n_iter = 1000,
    int32_t batch_size = 1,
    int device_id = 0
) {
    py::array J_arr = EnsureDenseFloat32C(J_obj);
    py::buffer_info J_buf = J_arr.request();
    if (J_buf.ndim != 2 || J_buf.shape[0] != J_buf.shape[1]) throw std::runtime_error("J must be square 2D");
    int32_t N = static_cast<int32_t>(J_buf.shape[0]);
    constexpr int32_t kAlign = 16;
    int32_t N_aligned = ((N + kAlign - 1) / kAlign) * kAlign;

    int32_t inferred_batch = batch_size;
    if (!x_obj.is_none()) {
        py::array x_tmp = EnsureFloat32Fortran2D(x_obj, N);
        inferred_batch = static_cast<int32_t>(x_tmp.shape(1));
    } else if (!y_obj.is_none()) {
        py::array y_tmp = EnsureFloat32Fortran2D(y_obj, N);
        inferred_batch = static_cast<int32_t>(y_tmp.shape(1));
    }

    py::array x_arr;
    py::array y_arr;
    const bool has_x = !x_obj.is_none();
    const bool has_y = !y_obj.is_none();
    if (!has_x && !has_y) {
        if (!TryGetLastSbInit(N, inferred_batch, &x_arr, &y_arr)) {
            x_arr = MakeRandomCenteredFloat32Fortran(N, inferred_batch);
            y_arr = MakeRandomCenteredFloat32Fortran(N, inferred_batch);
        }
    } else {
        x_arr = has_x ? EnsureFloat32Fortran2D(x_obj, N) : MakeRandomCenteredFloat32Fortran(N, inferred_batch);
        y_arr = has_y ? EnsureFloat32Fortran2D(y_obj, N) : MakeRandomCenteredFloat32Fortran(N, inferred_batch);
    }

    py::array p_arr;
    if (p_obj.is_none()) {
        p_arr = MakeLinspaceFloat32(n_iter);
    } else {
        py::module_ np = py::module_::import("numpy");
        py::array p_tmp = np.attr("asarray")(p_obj, py::arg("dtype") = np.attr("float32")).cast<py::array>();
        if (p_tmp.ndim() != 1) throw std::runtime_error("p must be 1D");
        if (p_tmp.shape(0) != n_iter) {
            throw std::runtime_error("p size must match n_iter");
        }
        p_arr = np.attr("ascontiguousarray")(p_tmp).cast<py::array>();
    }

    float xi = xi_obj.is_none() ? ComputeDefaultXiFromDenseJ(J_buf) : xi_obj.cast<float>();

    py::array J_use = J_arr;
    py::array x_use = x_arr;
    py::array y_use = y_arr;
    py::object h_use = h_obj;

    if (N_aligned != N) {
        py::array_t<float, py::array::c_style> J_pad({N_aligned, N_aligned});
        auto* J_pad_data = static_cast<float*>(J_pad.mutable_data());
        std::fill(J_pad_data, J_pad_data + static_cast<size_t>(N_aligned) * static_cast<size_t>(N_aligned), 0.0f);
        const float* J_data = static_cast<const float*>(J_buf.ptr);
        for (int32_t i = 0; i < N; ++i) {
            std::memcpy(J_pad_data + static_cast<size_t>(i) * static_cast<size_t>(N_aligned),
                        J_data + static_cast<size_t>(i) * static_cast<size_t>(N),
                        static_cast<size_t>(N) * sizeof(float));
        }

        py::array x_pad = MakeRandomCenteredFloat32Fortran(N_aligned, inferred_batch);
        py::array y_pad = MakeRandomCenteredFloat32Fortran(N_aligned, inferred_batch);
        std::fill(static_cast<float*>(x_pad.mutable_data()),
                  static_cast<float*>(x_pad.mutable_data()) + static_cast<size_t>(N_aligned) * static_cast<size_t>(inferred_batch),
                  0.0f);
        std::fill(static_cast<float*>(y_pad.mutable_data()),
                  static_cast<float*>(y_pad.mutable_data()) + static_cast<size_t>(N_aligned) * static_cast<size_t>(inferred_batch),
                  0.0f);

        const float* x_data = static_cast<const float*>(x_arr.request().ptr);
        const float* y_data = static_cast<const float*>(y_arr.request().ptr);
        float* x_pad_data = static_cast<float*>(x_pad.mutable_data());
        float* y_pad_data = static_cast<float*>(y_pad.mutable_data());
        for (int32_t b = 0; b < inferred_batch; ++b) {
            std::memcpy(x_pad_data + static_cast<size_t>(b) * static_cast<size_t>(N_aligned),
                        x_data + static_cast<size_t>(b) * static_cast<size_t>(N),
                        static_cast<size_t>(N) * sizeof(float));
            std::memcpy(y_pad_data + static_cast<size_t>(b) * static_cast<size_t>(N_aligned),
                        y_data + static_cast<size_t>(b) * static_cast<size_t>(N),
                        static_cast<size_t>(N) * sizeof(float));
        }

        if (!h_obj.is_none()) {
            py::array h_vec = EnsureFloat32CVector(h_obj, N);
            py::array_t<float, py::array::c_style> h_pad({N_aligned});
            auto* h_pad_data = static_cast<float*>(h_pad.mutable_data());
            std::fill(h_pad_data, h_pad_data + static_cast<size_t>(N_aligned), 0.0f);
            const float* h_data = static_cast<const float*>(h_vec.request().ptr);
            std::memcpy(h_pad_data, h_data, static_cast<size_t>(N) * sizeof(float));
            h_use = h_pad;
        }

        J_use = J_pad;
        x_use = x_pad;
        y_use = y_pad;
    }

    solve_bsb(
        J_use.cast<py::array_t<float, py::array::c_style>>(),
        x_use.cast<py::array_t<float, py::array::f_style>>(),
        y_use.cast<py::array_t<float, py::array::f_style>>(),
        delta,
        xi,
        dt,
        p_arr.cast<py::array_t<float, py::array::c_style>>(),
        n_iter,
        device_id,
        h_use);

    if (N_aligned != N) {
        py::module_ np = py::module_::import("numpy");
        py::array x_trim = np.attr("asfortranarray")(x_use.attr("__getitem__")(py::make_tuple(py::slice(0, N, 1), py::slice(0, inferred_batch, 1)))).cast<py::array>();
        py::array y_trim = np.attr("asfortranarray")(y_use.attr("__getitem__")(py::make_tuple(py::slice(0, N, 1), py::slice(0, inferred_batch, 1)))).cast<py::array>();
        return py::make_tuple(x_trim, y_trim);
    }

    return py::make_tuple(x_use, y_use);
}

PYBIND11_MODULE(kernel_sb, m) {
    m.doc() = "Kernel SB Ascend Plugin";
    m.def("solve", &solve, "Run Kernel DSB simulation",
          py::arg("J"), py::arg("x"), py::arg("y"),
          py::arg("delta"), py::arg("xi"), py::arg("dt"), 
          py::arg("p"), // Now array
          py::arg("niter"),
          py::arg("device_id") = 0,
          py::arg("h") = py::none());
    m.def("solve_cached", &solve_cached, "Run Kernel DSB simulation (reuses cached device buffers for J/p)",
          py::arg("J"), py::arg("x"), py::arg("y"),
          py::arg("delta"), py::arg("xi"), py::arg("dt"),
          py::arg("p"),
          py::arg("niter"),
          py::arg("force_copy") = false,
          py::arg("device_id") = 0,
          py::arg("h") = py::none());
    m.def("solve_bsb", &solve_bsb, "Run Kernel BSB simulation",
          py::arg("J"), py::arg("x"), py::arg("y"),
          py::arg("delta"), py::arg("xi"), py::arg("dt"),
          py::arg("p"),
          py::arg("niter"),
          py::arg("device_id") = 0,
          py::arg("h") = py::none());
    m.def("solve_cached_bsb", &solve_cached_bsb, "Run Kernel BSB simulation (reuses cached device buffers for J/p)",
          py::arg("J"), py::arg("x"), py::arg("y"),
          py::arg("delta"), py::arg("xi"), py::arg("dt"),
          py::arg("p"),
          py::arg("niter"),
          py::arg("force_copy") = false,
          py::arg("device_id") = 0,
          py::arg("h") = py::none());
    m.def("kernel_dsb", &kernel_dsb, "Run Kernel DSB with SB/DSB-style defaults; returns (x, y).",
          py::arg("J"),
          py::arg("h") = py::none(),
          py::arg("x") = py::none(),
          py::arg("y") = py::none(),
          py::arg("delta") = 1.0f,
          py::arg("xi") = py::none(),
          py::arg("dt") = 1.0f,
          py::arg("p") = py::none(),
          py::arg("n_iter") = 1000,
          py::arg("batch_size") = 1,
          py::arg("device_id") = 0);
    m.def("kernel_bsb", &kernel_bsb, "Run Kernel BSB with SB-style defaults; returns (x, y).",
          py::arg("J"),
          py::arg("h") = py::none(),
          py::arg("x") = py::none(),
          py::arg("y") = py::none(),
          py::arg("delta") = 1.0f,
          py::arg("xi") = py::none(),
          py::arg("dt") = 1.0f,
          py::arg("p") = py::none(),
          py::arg("n_iter") = 1000,
          py::arg("batch_size") = 1,
          py::arg("device_id") = 0);
}
