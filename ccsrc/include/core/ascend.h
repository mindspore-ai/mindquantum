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

inline std::string SocName(const char* target) {
    const auto* name = aclrtGetSocName();
    if (!name) {
        throw std::runtime_error("aclrtGetSocName returned no device model.");
    }
    if (std::string(name) != target) {
        throw std::runtime_error("Ascend kernel target " + std::string(target) + " does not match device " + name);
    }
    return name;
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
