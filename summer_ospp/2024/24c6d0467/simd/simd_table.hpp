# pragma once
#include <cstdint>
#include "simd/simd_word.hpp"
#include "simd/simd_array.hpp"
class SimdTable {

private:

    SimdWord* begin;

    size_t num_qbits;

    size_t major_nums_word;

    size_t minor_nums_word;

public:

    SimdTable(SimdWord* begin, size_t num_qbits, size_t major_nums_word, size_t minor_nums_word) : begin(begin), num_qbits(num_qbits), major_nums_word(major_nums_word), minor_nums_word(minor_nums_word) {}

    SimdArray operator[](const size_t q) {
        
        return SimdArray(begin + q * major_nums_word, minor_nums_word);
    }

    size_t get_num_qbits() {
        return num_qbits;
    }

    size_t get_major_nums_word() {
        return major_nums_word;
    }

    size_t get_minor_nums_word() {
        return minor_nums_word;
    }

    SimdWord* get_begin_simd() {
        return begin;
    }

    size_t get_index_of_word (size_t major_index_high, size_t major_index_low, size_t minor_index_high) const {
        size_t major_index = (major_index_high << SimdWord::BIT_POW) + major_index_low;
        return major_index * minor_nums_word + minor_index_high;
    }

    void exchange_low_indices(SimdTable &table) {
        for (size_t maj_high = 0; maj_high < table.get_major_nums_word(); maj_high++) {
            for (size_t min_high = 0; min_high < table.get_minor_nums_word(); min_high++) {
                size_t block_start = table.get_index_of_word(maj_high, 0, min_high);
                SimdWord::inplace_transpose_square(table.get_begin_simd() + block_start, table.get_minor_nums_word());
            }
        }
    }

    size_t get_index_of_bitword(
        size_t major_index_high, size_t major_index_low, size_t minor_index_high) const {
        size_t major_index = (major_index_high << SimdWord::BIT_POW) + major_index_low;
        return major_index * minor_nums_word + minor_index_high;
    }
    // 方阵转置
    void do_transposed() {
        assert(minor_nums_word == major_nums_word);
        exchange_low_indices(*this);

        // Current address tensor indices: [...maj_low ...min_high ...min_low ...maj_high]

        // Permute data such that high address bits of majors and minors are exchanged.
        for (size_t maj_high = 0; maj_high < major_nums_word; maj_high++) {
            for (size_t min_high = maj_high + 1; min_high < minor_nums_word; min_high++) {
                for (size_t maj_low = 0; maj_low < 256; maj_low++) {
                    std::swap(
                        begin[get_index_of_bitword(maj_high, maj_low, min_high)],
                        begin[get_index_of_bitword(min_high, maj_low, maj_high)]);
                }
            }
        }
    }

    void copy(SimdTable t) {
        assert(major_nums_word == t.major_nums_word);
        assert(minor_nums_word == t.minor_nums_word);
        for (size_t i = 0; i < major_nums_word; ++i) {
            for (size_t j = 0; j < minor_nums_word; ++j) {
                (*this)[i][j] = t[i][j];
            }
        }
    }

    void get_transposed(SimdWord* other_begin) {
        SimdTable result = SimdTable(other_begin, get_num_qbits(), get_major_nums_word(), get_minor_nums_word());
        for (size_t maj_high = 0; maj_high < get_major_nums_word(); maj_high++) {
            for (size_t min_high = 0; min_high < get_minor_nums_word(); min_high++) {
                for (size_t maj_low = 0; maj_low < 256; maj_low++) {
                    size_t src_index = result.get_index_of_word(maj_high, maj_low, min_high);
                    size_t dst_index = result.get_index_of_word(min_high, maj_low, maj_high);
                    result.get_begin_simd()[dst_index] = get_begin_simd()[src_index];
                }
            }
        }
        exchange_low_indices(result);
    }
};