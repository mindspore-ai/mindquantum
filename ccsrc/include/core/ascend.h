// Copyright 2026 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.
#ifndef INCLUDE_CORE_ASCEND_H_
#define INCLUDE_CORE_ASCEND_H_

#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>

#include <acl/acl.h>

namespace mindquantum::ascend {
inline void Check(aclError status, const char* operation) {
    if (status != ACL_SUCCESS) {
        const auto* detail = aclGetRecentErrMsg();
        const auto message = std::string("Ascend: ") + operation + " failed (" + std::to_string(status) + ")"
                             + (detail ? std::string(": ") + detail : "");
        // Runtime errors remain in thread-local storage after a checked API
        // returns them. Consume the error we report, or an unrelated later
        // kernel launch will observe it again despite succeeding.
        (void) aclrtGetLastError(ACL_RT_THREAD_LEVEL);
        throw std::runtime_error(message);
    }
}

inline void Cleanup(aclError status, const char* operation) noexcept {
    if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "Ascend: %s failed during cleanup (%d)\n", operation, status);
        (void) aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    }
}

inline void Initialize() {
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        const auto status = aclInit(nullptr);
        // Another library may own the runtime. Never finalize it or reset a device.
        if (status != ACL_ERROR_REPEAT_INITIALIZE) {
            Check(status, "aclInit");
        }
    });
}

// Preserve a caller-owned context, including on exceptions. CANN rejects
// SetCurrentContext(nullptr); previously unbound threads cannot be restored.
class ContextScope {
 public:
    explicit ContextScope(aclrtContext target = nullptr, void (*check)(aclError, const char*) = Check)
        : target_(target) {
        const auto status = aclrtGetCurrentContext(&previous_);
        if (status != ACL_ERROR_RT_CONTEXT_NULL) {
            check(status, "aclrtGetCurrentContext");
        }
        if (target_ && target_ != previous_) {
            check(aclrtSetCurrentContext(target_), "aclrtSetCurrentContext");
        }
    }
    ~ContextScope() {
        if (previous_ && previous_ != target_) {
            Cleanup(aclrtSetCurrentContext(previous_), "restore caller context");
        }
    }
    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;

 private:
    aclrtContext previous_ = nullptr;
    aclrtContext target_;
};

// Check before launching any kernel: CANN 9.1's ASC launch path can submit an
// incompatible binary and report an illegal instruction only at synchronization.
// Its SoC compatibility API also compares NPU architectures, but does not accept
// dav-* inputs; architecture builds query the device architecture directly.
inline std::string SocName([[maybe_unused]] int device) {
    const auto* name = aclrtGetSocName();
    if (!name) {
        throw std::runtime_error("aclrtGetSocName returned no device model.");
    }
    std::string device_name(name);
#ifdef MQ_ASCEND_ARCHITECTURE
    const auto* target = MQ_ASCEND_ARCHITECTURE;
    int64_t actual = 0;
    Check(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_NPU_ARCH, &actual), "aclrtGetDeviceInfo NPU architecture");
    const auto device_architecture = "dav-" + std::to_string(actual);
    const auto compatible = device_architecture == target;
    if (!compatible) {
        device_name += " (" + device_architecture + ")";
    }
#else
    const auto* target = MQ_ASCEND_SOC;
    int32_t compatible = 0;
    Check(aclrtCheckArchCompatibility(target, &compatible), "aclrtCheckArchCompatibility");
#endif
    if (!compatible) {
        throw std::runtime_error("Ascend kernel target " + std::string(target) + " does not match device "
                                 + device_name);
    }
    return device_name;
}

// Declare after the stream's input owners, inside their context/lock scope.
// Drain failed asynchronous work before freeing or reusing its host buffers.
class Transaction {
 public:
    explicit Transaction(aclrtStream stream) : stream_(stream) {
    }
    ~Transaction() {
        if (!finished_) {
            Cleanup(aclrtSynchronizeStream(stream_), "drain failed transaction");
        }
    }
    void Finish(const char* operation) {
        Check(aclrtSynchronizeStream(stream_), operation);
        finished_ = true;
    }
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;

 private:
    aclrtStream stream_;
    bool finished_ = false;
};
}  // namespace mindquantum::ascend
#endif  // INCLUDE_CORE_ASCEND_H_
