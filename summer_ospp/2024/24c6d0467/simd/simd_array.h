#include <cstdint>
#include <array>
#include <random>
#include <immintrin.h>
#include <assert.h>
#include "utils.h"
#include "bit.h"
#include "simd_word.h"
using namespace std;
// 實現一個SIMD指令單元接口
struct SimdArray {
    // 数组长度（根据）
    size_t num_bit_words;

    SimdWord *ptr_simd;

    public:

        // 256位AVX指令， 一個uint16占4字节 256位占32字节
        SimdArray(size_t num_bit_words) : num_bit_words(num_bit_words) {
        };
        SimdArray(SimdWord *ptr_simd, size_t num_bit_words) : ptr_simd(ptr_simd) , num_bit_words(num_bit_words){};

        SimdArray() {
        };

        SimdWord* operator[](size_t index) {
            assert(index < num_bit_words && index >=0);
            return ptr_simd + index;
        }
        // 获取SIMD指令中的某一个bit
        bit get_bit(size_t k) {
            return bit(ptr_simd, k);
        }
        template<typename FUNC>
        void for_each_word(FUNC func) {
            SimdWord *now = ptr_simd;
            SimdWord *end = ptr_simd + num_bit_words;
            while (now != end) {
                func(*now);
                now++;
            } 
        }

        template<typename FUNC>
        void for_each_word(SimdArray other, FUNC body) {
            
        }

        template<typename FUNC>
        void for_each_word(SimdArray other1, SimdArray other2, FUNC body) {

        }

        template<typename FUNC>
        void for_each_word(const SimdArray *other1, const SimdArray *other2, const SimdArray* other3, FUNC body) {
            auto *begin = ptr_simd;
            auto *begin1 = other1->ptr_simd;
            auto *begin2 = other2->ptr_simd;
            auto *begin3 = other3->ptr_simd;
            auto *end = ptr_simd + num_bit_words;
            while (begin != end) {
                body(*begin, *begin1, *begin2, *begin3);
                begin++;
                begin1++;
                begin2++;
                begin3++;
            }
        };

        SimdArray operator^(SimdArray &other) {
            return {};
        }

        SimdArray operator^=(SimdWord &other) {
            
        };
    private:
        __m256i initialize_m256_by_array(array<uint16_t, 4> val) {
            
        }
};