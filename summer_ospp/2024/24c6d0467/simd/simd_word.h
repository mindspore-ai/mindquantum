#include<iostream>
#include<array>
#include<immintrin.h>
#include <cstdint>
#include<bit>
#include<bitset>
#include<initializer_list>
#include "utils.h"
constexpr size_t min_bits_to_num_bits_padded(size_t min_bits) {
// 计算逻辑解析：
    return (min_bits + (sizeof(SimdWord) * 8 - 1)) & ~(sizeof(SimdWord) * 8 - 1);
}

// 将需要的比特数转换成simd指令数
constexpr size_t min_bits_to_num_simd_words(size_t min_bits) {
    return (min_bits_to_num_bits_padded(min_bits) / sizeof(SimdWord)) >> 3;
}
    
// 根据比特数malloc指定大小
template <size_t W>
uint64_t *malloc_aligned_padded_zeroed(size_t min_bits) {
    size_t num_u8 = min_bits_to_num_bits_padded(min_bits) >> 3;
    void *result = SimdWord::aligned_malloc(num_u8);
    memset(result, 0, num_u8);
    return (uint64_t *)result;
}

struct SimdWord {
    // simd指令最小单元
    // 这里在实现
    __m256i val;

public:

    SimdWord() {}

    SimdWord(__m256i val): val(val) {}

    SimdWord(std::array<uint16_t, 16> val): val(initialize_m256_by_array(val)) {}

    SimdWord(const SimdWord& other) : val(other.val){}

    static void *aligned_malloc(size_t bytes) {
        return _mm_malloc(bytes, sizeof(__m256i));            
    }
    __m256i initialize_m256_by_array(std::array<uint16_t, 16> val) {
        _mm256_set_epi16(val[15], val[14], val[13], val[12], 
        val[11], val[10], val[9], val[8],
        val[7], val[6], val[5], val[4],
        val[3], val[2], val[1], val[0]);}

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

    std::array<uint16_t, 16> to_u16_array() {
        std::array<uint16_t, 16> result;
        for (auto i : result) {
            result[i] = _mm256_extract_epi16(val, i);
        }
        return result;
    }


    uint16_t popcount(){
        auto arr = to_u16_array();
        uint16_t count;
        for(auto i : arr) {
            count += utils::popcount(arr[i]);
        }
        return c
        ount;
    }
};