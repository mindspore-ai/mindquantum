//   Copyright 2021 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#ifndef ALIGNED_ALLOCATOR_HPP
#define ALIGNED_ALLOCATOR_HPP

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

// NOLINTNEXTLINE
#define DEFAULT_ALIGNMENT 16U

#ifdef _WIN32
#    include <malloc.h>
#endif

#ifndef MALLOC_ALREADY_ALIGNED
#    if defined(__GLIBC__) && ((__GLIBC__ >= 2 && __GLIBC_MINOR__ >= 8) || __GLIBC__ > 2) && defined(__LP64__)         \
        && !defined(__SANITIZE_ADDRESS__)
#        define GLIBC_MALLOC_ALREADY_ALIGNED 1  // NOLINT
#    else
#        define GLIBC_MALLOC_ALREADY_ALIGNED 0  // NOLINT
#    endif

// FreeBSD 6 seems to have 16-byte aligned malloc
//   See http://svn.freebsd.org/viewvc/base/stable/6/lib/libc/stdlib/malloc.c?view=markup
// FreeBSD 7 seems to have 16-byte aligned malloc except on ARM and MIPS architectures
//   See http://svn.freebsd.org/viewvc/base/stable/7/lib/libc/stdlib/malloc.c?view=markup
#    if defined(__FreeBSD__) && !(defined(__arm__) || defined(__mips__) || defined(__mips))
#        define FREEBSD_MALLOC_ALREADY_ALIGNED 1  // NOLINT
#    else
#        define FREEBSD_MALLOC_ALREADY_ALIGNED 0  // NOLINT
#    endif

#    if (defined(__APPLE__) || defined(_WIN64) || GLIBC_MALLOC_ALREADY_ALIGNED || FREEBSD_MALLOC_ALREADY_ALIGNED)
#        define MALLOC_ALREADY_ALIGNED 1  // NOLINT
#    else
#        define MALLOC_ALREADY_ALIGNED 0  // NOLINT
#    endif
#endif  // !MALLOC_ALREADY_ALIGNED

#if ((defined __QNXNTO__) || (defined _GNU_SOURCE) || ((defined _XOPEN_SOURCE) && (_XOPEN_SOURCE >= 600)))             \
    && (defined _POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO > 0)
#    define HAS_POSIX_MEMALIGN 1  // NOLINT
#else
#    define HAS_POSIX_MEMALIGN 0  // NOLINT
#endif

#if (defined(_M_AMD64) || defined(_M_X64) || defined(__amd64)) && !defined(__x86_64__)
#    define __x86_64__ 1  // NOLINT
#endif

// Find sse instruction set from compiler macros if SSE_INSTR_SET not defined
// Note: Not all compilers define these macros automatically
#ifndef SSE_INSTR_SET
#    if defined(__AVX2__)
#        define SSE_INSTR_SET 8  // NOLINT
#    elif defined(__AVX__)
#        define SSE_INSTR_SET 7  // NOLINT
#    elif defined(__SSE4_2__)
#        define SSE_INSTR_SET 6  // NOLINT
#    elif defined(__SSE4_1__)
#        define SSE_INSTR_SET 5  // NOLINT
#    elif defined(__SSSE3__)
#        define SSE_INSTR_SET 4  // NOLINT
#    elif defined(__SSE3__)
#        define SSE_INSTR_SET 3  // NOLINT
#    elif defined(__SSE2__) || defined(__x86_64__)
#        define SSE_INSTR_SET 2  // NOLINT
#    elif defined(__SSE__)
#        define SSE_INSTR_SET 1  // NOLINT
#    elif defined(_M_IX86_FP)    // Defined in MS compiler on 32bits system. 1: SSE, 2: SSE2
#        define SSE_INSTR_SET _M_IX86_FP
#    else
#        define SSE_INSTR_SET 0  // NOLINT
#    endif                       // instruction set defines
#endif                           // SSE_INSTR_SET

#if SSE_INSTR_SET > 0
#    define HAS_MM_MALLOC 1  // NOLINT
#    include <xmmintrin.h>
#else
#    define HAS_MM_MALLOC 0  // NOLINT
#endif

#if __cplusplus >= 201703
#    define HAS_STD_ALIGNED_ALLOC 1  // NOLINT
#else
#    define HAS_STD_ALIGNED_ALLOC 0  // NOLINT
#endif

#if __cplusplus < 201103L
#    error aligned_allocator requires at least C++11 support enabled!
#endif

template <class T, unsigned int alignment>
class aligned_allocator : public std::allocator<T> {
 public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    static_assert(alignment >= DEFAULT_ALIGNMENT, "Alignment must be equal or greater than default alignment");

    template <class U>
    struct rebind {
        using other = aligned_allocator<U, alignment>;
    };

    aligned_allocator() noexcept = default;
    aligned_allocator(const aligned_allocator& /* unused */) noexcept = default;
    aligned_allocator(aligned_allocator&& /* unused */) noexcept = default;

    template <class U>
    explicit aligned_allocator(const aligned_allocator<U, alignment>& /* unused */) noexcept {
    }

    ~aligned_allocator() noexcept = default;
    aligned_allocator& operator=(const aligned_allocator&) noexcept = default;
    aligned_allocator& operator=(aligned_allocator&&) noexcept = default;

    // NOLINTNEXTLINE(huawei-force-type-void)
    auto allocate(size_type n, const void* /* hint */ = nullptr) const;
    void deallocate(pointer p, size_type /* unused */) const;
};

namespace detail {
// NOLINTNEXTLINE(huawei-force-type-void)
inline void* _aligned_malloc(size_t size, size_t alignment) {
    void* res = nullptr;
    void* ptr = malloc(size + alignment);  // NOLINT
    if (ptr != nullptr) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        res = reinterpret_cast<void*>((reinterpret_cast<size_t>(ptr) & ~(size_t(alignment - 1))) + alignment);
        *(reinterpret_cast<void**>(res) - 1) = ptr;  // NOLINT
    }
    return res;
}
}  // namespace detail

// NOLINTNEXTLINE(huawei-force-type-void)
inline void* aligned_malloc(size_t size, size_t alignment) {  // NOLINT(misc-unused-parameters)
#if MALLOC_ALREADY_ALIGNED
    return malloc(size);  // NOLINT
#elif HAS_MM_MALLOC
    return _mm_malloc(size, alignment);  // NOLINT
#elif HAS_POSIX_MEMALIGN
    void* res;
    const int failed = posix_memalign(&res, alignment, size);
    if (failed) {
        res = nullptr;
    }
    return res;
#elif (defined _MSC_VER)
    return _aligned_malloc(size, alignment);
#elif HAS_STD_ALIGNED_ALLOC
    return std::aligned_alloc(alignment, size);
#else
    return detail::_aligned_malloc(size, alignment);
#endif
}

namespace detail {
// NOLINTNEXTLINE(huawei-force-type-void)
inline void _aligned_free(void* ptr) {
    if (ptr != nullptr) {
        free(*(reinterpret_cast<void**>(ptr) - 1));  // NOLINT
    }
}
}  // namespace detail

// NOLINTNEXTLINE(huawei-force-type-void)
inline void aligned_free(void* ptr) {
#if MALLOC_ALREADY_ALIGNED
    free(ptr);  // NOLINT
#elif HAS_MM_MALLOC
    _mm_free(ptr);                       // NOLINT
#elif HAS_POSIX_MEMALIGN
    free(ptr);  // NOLINT
#elif defined(_MSC_VER)
    _aligned_free(ptr);  // NOLINT
#elif HAS_STD_ALIGNED_ALLOC
    std::free(ptr);  // NOLINT
#else
    detail::_aligned_free(ptr);  // NOLINT
#endif
}

template <class T, unsigned int alignment>
// NOLINTNEXTLINE(huawei-force-type-void)
auto aligned_allocator<T, alignment>::allocate(size_type n, const void* /* hint */) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto res = reinterpret_cast<pointer>(aligned_malloc(sizeof(T) * n, alignment));
    if (res == nullptr) {
        throw std::bad_alloc();
    }
    return res;
}

template <class T, unsigned int alignment>
void aligned_allocator<T, alignment>::deallocate(pointer p, size_type /* unused */) const {
    aligned_free(p);
}
#endif /* ALIGNED_ALLOCATOR_HPP */
