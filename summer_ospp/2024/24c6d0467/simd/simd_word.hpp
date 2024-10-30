#pragma once
#include <iostream>
#include <array>
#include <immintrin.h>
#include <cstdint>
#include <bit>
#include <bitset>
#include <initializer_list>
#include <cstring>
// void *aligned_malloc(size_t bytes) {
//         return _mm_malloc(bytes, sizeof(__m256i));            
// }

// 将需要的比特数转换成simd指令数

struct SimdWord {
    const static size_t BIT_POW = 8;
    // simd指令最小单元
    // 这里在实现
    __m256i val;

public:

    SimdWord() : val(__m256i{}) {}

    SimdWord(__m256i val): val(val) {}

    SimdWord(std::array<uint64_t, 4> val): val(initialize_m256_by_array(val)) {}

    SimdWord(const SimdWord& other) : val(other.val){}

    __m256i initialize_m256_by_array(std::array<uint64_t, 4> val) {
        return _mm256_set_epi64x(val[3], val[2], val[1], val[0]);}

    SimdWord &operator^=(const SimdWord &other) {
        val = _mm256_xor_si256(val, other.val);
        return *this;
    }
    
    SimdWord operator^(const SimdWord& other) {
        return {_mm256_xor_si256(val, other.val)};
    }
    
    SimdWord operator&(const SimdWord& other) {
        return {_mm256_and_si256(val, other.val)};
    }

    SimdWord operator|(const SimdWord &other) const {
        return {_mm256_or_si256(val, other.val)};
    }

    SimdWord andnot(const SimdWord &other) const {
        return {_mm256_andnot_si256(val, other.val)};
    }
    
    std::array<uint64_t, 4> to_u64_array() const{
        return std::array<uint64_t, 4>{
            (uint64_t) _mm256_extract_epi64(val, 0),
            (uint64_t) _mm256_extract_epi64(val, 1),
            (uint64_t) _mm256_extract_epi64(val, 2),
            (uint64_t) _mm256_extract_epi64(val, 3),
        };
    }

    SimdWord& operator|=(const SimdWord& other) {
        val = _mm256_or_si256(val, other.val);
        return *this;
    }

    operator bool() const{
        auto word = to_u64_array();
        return (bool)(word[0] | word[1] | word[2] | word[3]);
    }

    uint16_t popcount(uint16_t num) {
        uint16_t count = 0;
        while (num) {
            count += num & 1;
            num >>= 1;
        }
        return count;
    }

    uint16_t popcount() {
        auto arr = to_u64_array();
        uint16_t count;
        
        for(auto i : arr) {
            count += popcount(i);
        }
        return count;
    }

    template <uint64_t shift>
    static void inplace_transpose_block_pass(SimdWord *data, size_t stride, __m256i mask) {
        for (size_t k = 0; k < 256; k++) {
            if (k & shift) {
                continue;
            }
            SimdWord &x = data[stride * k];
            SimdWord &y = data[stride * (k + shift)];
            SimdWord a = x & mask;
            SimdWord b = x & ~mask;
            SimdWord c = y & mask;
            SimdWord d = y & ~mask;
            x = a | SimdWord(_mm256_slli_epi64(c.val, shift));
            y = SimdWord(_mm256_srli_epi64(b.val, shift)) | d;
        }
    }
    
    static void inplace_transpose_block_pass_64_and_128(SimdWord* data, size_t stride) {
        uint64_t *ptr = (uint64_t *)data;
        stride <<= 2;
        for (size_t k = 0; k < 64; k++) {
            std::swap(ptr[stride * (k + 64 * 0) + 1], ptr[stride * (k + 64 * 1) + 0]);
            std::swap(ptr[stride * (k + 64 * 0) + 2], ptr[stride * (k + 64 * 2) + 0]);
            std::swap(ptr[stride * (k + 64 * 0) + 3], ptr[stride * (k + 64 * 3) + 0]);
            std::swap(ptr[stride * (k + 64 * 1) + 2], ptr[stride * (k + 64 * 2) + 1]);
            std::swap(ptr[stride * (k + 64 * 1) + 3], ptr[stride * (k + 64 * 3) + 1]);
            std::swap(ptr[stride * (k + 64 * 2) + 3], ptr[stride * (k + 64 * 3) + 2]);
        }
    }

    void to_str() {
        std::cout << "bitword<" <<">{";
        auto u = to_u64_array();
        for (size_t k = 0; k < u.size(); k++) {
            for (size_t b = 0; b < 64; b++) {
                if ((b | k) && (b & 7) == 0) {
                    std::cout << ' ';
                }
                std::cout << ".1"[(u[k] >> b) & 1];
            }
        }
        std::cout << '}';
    }

    void print_arr() {
        auto u = to_u64_array();
        for(int i = 0; i < 4; ++i) {
            std::cout << u[i] << " ";
        }
        std::cout << std::endl;
    }
    static void inplace_transpose_square(SimdWord *data, size_t stride) {
        inplace_transpose_block_pass<1>(data, stride, _mm256_set1_epi8(0x55));
        inplace_transpose_block_pass<2>(data, stride, _mm256_set1_epi8(0x33));
        inplace_transpose_block_pass<4>(data, stride, _mm256_set1_epi8(0xF));
        inplace_transpose_block_pass<8>(data, stride, _mm256_set1_epi16(0xFF));
        inplace_transpose_block_pass<16>(data, stride, _mm256_set1_epi32(0xFFFF));
        inplace_transpose_block_pass<32>(data, stride, _mm256_set1_epi64x(0xFFFFFFFF));
        inplace_transpose_block_pass_64_and_128(data, stride);
    }
};