// Copyright 2026 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.
#ifndef INCLUDE_CORE_ASCEND_KERNEL_LIBRARY_H_
#define INCLUDE_CORE_ASCEND_KERNEL_LIBRARY_H_

#include <dlfcn.h>

#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <stdexcept>
#include <string>

#include "core/ascend.h"

namespace mindquantum::ascend {
// Select before loading/registering kernels, not after a failed launch. CANN's
// SoC compatibility API does not accept dav-* architecture identifiers.
inline std::string KernelTarget([[maybe_unused]] int device) {
#ifdef MQ_ASCEND_ARCHITECTURES
    int64_t actual = 0;
    Check(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_NPU_ARCH, &actual), "aclrtGetDeviceInfo NPU architecture");
    const auto architecture = "dav-" + std::to_string(actual);
    const char* targets[] = {MQ_ASCEND_ARCHITECTURES};
    std::string available;
    for (const auto* target : targets) {
        if (architecture == target) {
            return architecture;
        }
        available += (available.empty() ? "" : ", ") + std::string(target);
    }
    throw std::runtime_error("Ascend device " + SocName() + " (" + architecture
                             + ") has no kernels in this build; available targets: " + available);
#else
    int32_t compatible = 0;
    Check(aclrtCheckArchCompatibility(MQ_ASCEND_SOC, &compatible), "aclrtCheckArchCompatibility");
    if (!compatible) {
        throw std::runtime_error("Ascend kernel target " + std::string(MQ_ASCEND_SOC) + " does not match device "
                                 + SocName());
    }
    return MQ_ASCEND_SOC;
#endif
}

// CANN registers process-wide profiling callbacks into the kernel library.
// Keep its code and static tables mapped until process exit, independently of
// device resources. Contexts, streams and buffers still have explicit owners.
template <typename Api>
const Api* LoadKernels(int device, const char* library, const void* anchor) {
    const auto target = KernelTarget(device);
    Dl_info info{};
    if (!dladdr(anchor, &info)) {
        throw std::runtime_error("Cannot locate the Ascend host library");
    }
    const auto path = std::filesystem::path(info.dli_fname).parent_path() / "lib" / "ascend"
                      / (std::string("lib") + library + "_" + target + ".so");
    std::unique_ptr<void, int (*)(void*)> handle(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE), dlclose);
    if (!handle) {
        const auto* error = dlerror();
        throw std::runtime_error("Cannot load Ascend kernels " + path.string() + ": "
                                 + (error ? error : "dlopen returned null"));
    }
    dlerror();
    const auto entry = reinterpret_cast<const Api* (*)()>(dlsym(handle.get(), "mq_ascend_kernels_v1"));
    if (const auto* error = dlerror()) {
        throw std::runtime_error("Invalid Ascend kernel library " + path.string() + ": " + error);
    }
    if (!entry) {
        throw std::runtime_error("Invalid Ascend kernel library " + path.string() + ": null entry point");
    }
    const auto* api = entry();
    if (!api) {
        throw std::runtime_error("Invalid Ascend kernel library " + path.string() + ": null kernel table");
    }
    return api;
}
}  // namespace mindquantum::ascend
#endif  // INCLUDE_CORE_ASCEND_KERNEL_LIBRARY_H_
