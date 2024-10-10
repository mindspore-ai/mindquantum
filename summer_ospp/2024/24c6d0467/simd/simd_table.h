#include<iostream>
#include "simd_array.h"
#include "pauli_string.h"
#include "utils.h"
#include "simd_word.h"
// 每次操作肯定都是一行一行的操作，所以不会存在
class SimdTable{
    //table 理解为一个simd_word的二维数组

    // major 即是他的索引长度
    size_t major_nums;
    //宽 第二个宽度
    size_t minor_nums;
    //数据
    SimdArray data;
    SimdArray* ptr;

    public:
        SimdTable(size_t major_index, size_t minor_index) : major_nums(major_index), minor_nums(minor_index), data(SimdArray(min_bits_to_num_bits_padded(major_nums) * min_bits_to_num_bits_padded(minor_nums))) {}

        SimdTable(size_t major_nums, size_t minor_nums, SimdArray other) : data(other) {}

        // 重载[]运算符
        SimdArray* operator[](size_t index) {
            assert(index < major_nums);
            return ptr + index;
        }

};