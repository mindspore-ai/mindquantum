#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include "acl/acl.h"
#include "aclrtlaunch_kernel_sb_custom.h"
#include "data_utils.h"

constexpr int32_t N = 1024;
constexpr int32_t BATCH_SIZE = 16; // Example batch size for test
constexpr int32_t DEVICE_ID = 0;
constexpr int32_t NITER = 100;

int main() {
    // 1. Init ACL
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(DEVICE_ID));
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    // 2. Prepare Host Data
    std::vector<float> h_J(N * N);
    std::vector<float> h_x(N * BATCH_SIZE);
    std::vector<float> h_y(N * BATCH_SIZE);
    std::vector<float> h_p(NITER);
    std::vector<float> h_h(N, 0.0f);

    // Init J with standard random
    InitRandom(h_J, -0.1f, 0.1f);
    // Init x with new centered random logic: 0.02 * (rand - 0.5)
    InitRandomCentered(h_x, 0.02f);
    
    // Init y with standard random
    // InitRandom(h_y, -0.5f, 0.5f);
    InitRandomCentered(h_y, 0.02f);
    
    // Init p with linspace(0, 1)
    for (int i = 0; i < NITER; ++i) {
        h_p[i] = (float)i / (NITER - 1);
    }

    std::cout << "Initial x (first 10): "; PrintSlice(h_x, 10);
    std::cout << "Initial y (first 10): "; PrintSlice(h_y, 10);

    // 3. Prepare Device Memory
    void *d_J, *d_x, *d_y, *d_p, *d_h;
    size_t J_size = N * N * sizeof(float);
    size_t vec_size = N * BATCH_SIZE * sizeof(float);
    size_t p_size = NITER * sizeof(float);
    size_t h_size = N * sizeof(float);

    CHECK_ACL(aclrtMalloc(&d_J, J_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_x, vec_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_y, vec_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_p, p_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_h, h_size, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(d_J, J_size, h_J.data(), J_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_x, vec_size, h_x.data(), vec_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_y, vec_size, h_y.data(), vec_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_p, p_size, h_p.data(), p_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_h, h_size, h_h.data(), h_size, ACL_MEMCPY_HOST_TO_DEVICE));

    // 4. Parameters
    float delta = 1.0f;
    float xi = 1.0f;
    float dt = 0.1f;
    int32_t has_h = 0;

    // 5. Launch Kernel
    // Use BATCH_SIZE as blockDim
    aclrtlaunch_kernel_sb_custom(BATCH_SIZE, stream, d_J, d_x, d_y, d_p, d_h, has_h, delta, xi, dt, NITER, N, BATCH_SIZE);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    // 6. Retrieve Result
    CHECK_ACL(aclrtMemcpy(h_x.data(), vec_size, d_x, vec_size, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(h_y.data(), vec_size, d_y, vec_size, ACL_MEMCPY_DEVICE_TO_HOST));

    std::cout << "Final x (first 10): "; PrintSlice(h_x, 10);
    std::cout << "Final y (first 10): "; PrintSlice(h_y, 10);

    // 7. Cleanup
    CHECK_ACL(aclrtFree(d_J));
    CHECK_ACL(aclrtFree(d_x));
    CHECK_ACL(aclrtFree(d_y));
    CHECK_ACL(aclrtFree(d_p));
    CHECK_ACL(aclrtFree(d_h));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(DEVICE_ID));
    CHECK_ACL(aclFinalize());

    return 0;
}
