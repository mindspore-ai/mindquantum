#pragma once
#include "simd/simd_word.hpp"
#include "simd/bit.hpp"
#include <assert.h>

class SimdArray {

private:

    SimdWord* ptr_simd;

    size_t num_words;

public:

    SimdArray(SimdWord* ptr_simd, size_t num_words) : ptr_simd(ptr_simd), num_words(num_words) {}

    bit operator[](size_t index) {
        return bit(ptr_simd, index);
    }

    SimdWord* get_word_of_index(size_t index) {
        assert(index >= 0 && index <= num_words);
        return ptr_simd + index;
    }

    template<typename FUNC>
    void for_each_word(FUNC func) {
        SimdWord *now = ptr_simd;
        SimdWord *end = ptr_simd + num_words;
        while (now != end) {
            func(*now);
            now++;
        } 
    }

    template<typename FUNC>
    void for_each_word(SimdArray other, FUNC body) {
        auto *v0 = ptr_simd;
        auto *v1 = other.ptr_simd;
        auto *v0_end = v0 + num_words;
        while (v0 != v0_end) {
            body(*v0, *v1);
            v0++;
            v1++;
        }
    }

    template<typename FUNC>
    void for_each_word(SimdArray other1, SimdArray other2, FUNC body) {
        auto *v0 = ptr_simd;
        auto *v1 = other1.ptr_simd;
        auto *v2 = other2.ptr_simd;
        auto *v0_end = v0 + num_words;
        while (v0 != v0_end) {
            body(*v0, *v1, *v2);
            v0++;
            v1++;
            v2++;
        }
    }

    template<typename FUNC>
    void for_each_word(const SimdArray other1, const SimdArray other2, const SimdArray other3, FUNC body) {
        auto *begin = ptr_simd;
        auto *begin1 = other1.ptr_simd;
        auto *begin2 = other2.ptr_simd;
        auto *begin3 = other3.ptr_simd;
        auto *end = ptr_simd + num_words;
        while (begin != end) {
            body(*begin, *begin1, *begin2, *begin3);
            begin++;
            begin1++;
            begin2++;
            begin3++;
        }
    }

    template<typename FUNC>
    void for_each_word(const SimdArray other1, const SimdArray other2, const SimdArray other3, const SimdArray other4, FUNC body) {
        auto *begin = ptr_simd;
        auto *begin1 = other1.ptr_simd;
        auto *begin2 = other2.ptr_simd;
        auto *begin3 = other3.ptr_simd;
        auto *begin4 = other4.ptr_simd;
        auto *end = ptr_simd + num_words;
        while (begin != end) {
            body(*begin, *begin1, *begin2, *begin3, *begin4);
            begin++;
            begin1++;
            begin2++;
            begin3++;
            begin4++; 
        }
    }

    SimdArray& operator^=(SimdArray &other) {
        for_each_word(other, [](SimdWord& w0, SimdWord &w1) {
        w0 ^= w1;
        });
        return *this;
    };

    void operator^(SimdArray &other) {
        for_each_word(other, [](SimdWord& w0, SimdWord &w1) {
        w0 ^= w1;
        });
    }
    
    void swap_with(SimdArray &other) {
        for_each_word(other, [](SimdWord &w0, SimdWord &w1) {
        std::swap(w0, w1);
        });
    }

    const bool not_zero() {
        SimdWord acc{};
        for_each_word([&acc](SimdWord &w) {
        acc |= w;
        });
        return (bool)acc;
    }
};