#include <cstdint>
#include <immintrin.h>
#include "simd/simd_table.hpp"
#include "simd/simd_array.hpp"
#include "simd/simd_word.hpp"
#include "tableau/pauli_string.hpp"

inline size_t min_bits_to_num_bits_padded(size_t min_bits) {
    // 计算逻辑解析：
    return (min_bits + (sizeof(SimdWord) * 8 - 1)) & ~(sizeof(SimdWord) * 8 - 1);
};

inline size_t min_bits_to_num_simd_words(size_t min_bits) {
    return (min_bits_to_num_bits_padded(min_bits) / sizeof(SimdWord) ) >> 3;
};

inline size_t num_bits_padded_to_num_simd_words(size_t num_bits) {
    return (num_bits / sizeof(SimdWord)) >> 3;
};

inline SimdWord* malloc_aligned_padded_zeroed(size_t min_bits) {
    size_t num_u8 = min_bits_to_num_bits_padded(min_bits) >> 3;
    void *result = _mm_malloc(num_u8, sizeof(__m256i));
    memset(result, 0, num_u8);
    return (SimdWord *) result;
};

class TableauHalf {

private:

    size_t num_qbits;

    size_t table_major_nums_words;

    size_t table_minor_nums_words;

    size_t signs_nums_words;

    SimdWord* ptr_xt;

    SimdWord* ptr_zt;

    SimdWord* ptr_signs;

public:

    TableauHalf(size_t q) : num_qbits(q), table_major_nums_words(min_bits_to_num_simd_words(q)), table_minor_nums_words(min_bits_to_num_simd_words(q)), signs_nums_words(min_bits_to_num_bits_padded(min_bits_to_num_bits_padded(q))),
    ptr_xt(malloc_aligned_padded_zeroed(min_bits_to_num_bits_padded(q) * min_bits_to_num_bits_padded(q))), ptr_zt(malloc_aligned_padded_zeroed(min_bits_to_num_bits_padded(q) * min_bits_to_num_bits_padded(q))),
    ptr_signs(malloc_aligned_padded_zeroed(min_bits_to_num_bits_padded(q))) {}

    TableauHalf(const TableauHalf& tableau_half) : TableauHalf(tableau_half.get_q()) {
        auto xt = get_x_table();
        auto zt = get_z_table();
        xt.copy(tableau_half.get_x_table());
        zt.copy(tableau_half.get_z_table());
    }

    TableauHalf(size_t num_qbits, size_t table_major_nums_words, size_t table_minor_nums_words, size_t signs_nums_words,SimdWord* ptr_xt, SimdWord* ptr_zt, SimdWord* ptr_signs) 
    : num_qbits(num_qbits), table_major_nums_words(table_major_nums_words), table_minor_nums_words(table_minor_nums_words), signs_nums_words(signs_nums_words), ptr_xt(ptr_xt), ptr_zt(ptr_zt), ptr_signs(ptr_signs) {}

    ~TableauHalf() {
        if (ptr_zt != nullptr) {
            _mm_free(ptr_zt);
        }
        if (ptr_xt != nullptr) {
            _mm_free(ptr_xt);
        }
        if (ptr_signs != nullptr) {
            _mm_free(ptr_signs);
        }
    }

    PauliString operator[](size_t q) const{
        return PauliString(num_qbits, q, table_minor_nums_words, ptr_xt, ptr_zt, ptr_signs);
    }

    SimdTable get_x_table() const{
        return SimdTable(ptr_xt, num_qbits, table_major_nums_words, table_minor_nums_words);
    }
    
    SimdTable get_z_table() const{
        return SimdTable(ptr_zt, num_qbits, table_major_nums_words, table_minor_nums_words);
    }
    
    SimdArray get_signs() const{
        return SimdArray(ptr_signs, signs_nums_words);
    }

    size_t get_q() const{
        return num_qbits;
    }

    size_t get_table_major_nums_word() const{
        return table_major_nums_words;
    }

    size_t get_table_minor_nums_word() const{
        return table_minor_nums_words;
    }

    size_t get_signs_word() const{
        return signs_nums_words;
    }

    // static TableauHalf get_transposed(TableauHalf& tableau_half) {
    //     SimdWord* ptr_xt = malloc_aligned_padded_zeroed(min_bits_to_num_bits_padded(tableau_half.get_q()) * min_bits_to_num_bits_padded(tableau_half.get_q()));
    //     SimdWord* ptr_zt = malloc_aligned_padded_zeroed(min_bits_to_num_bits_padded(tableau_half.get_q()) * min_bits_to_num_bits_padded(tableau_half.get_q()));
    //     SimdWord* ptr_signs = malloc_aligned_padded_zeroed(min_bits_to_num_bits_padded(tableau_half.get_q()));
    //     tableau_half.get_x_table().get_transposed(ptr_xt);
    //     tableau_half.get_z_table().get_transposed(ptr_zt);
    //     return TableauHalf(tableau_half.get_q(), ptr_signs, ptr_xt, ptr_zt, tableau_half.get_table_minor_nums_word(), tableau_half.get_table_major_nums_word(), tableau_half.get_signs_word());
    // }
};

class Tableau {

private:

    size_t num_qbits;

    TableauHalf xs;

    TableauHalf zs;

public:

    Tableau(const Tableau& tableau) : num_qbits(tableau.get_q()), xs(TableauHalf(tableau.get_xs())), zs(TableauHalf(tableau.get_zs())) {}

    Tableau(size_t num_qbits) : num_qbits(num_qbits), xs(num_qbits), zs(num_qbits) {
        for (size_t i = 0; i < num_qbits; ++i) {
            xs.get_x_table()[i][i] = true;
            zs.get_z_table()[i][i] = true;
        }
    }

    Tableau(size_t num_qbits, TableauHalf xs, TableauHalf zs) : num_qbits(num_qbits), xs(xs), zs(zs) {}

    void do_X(size_t q) {
        for (int i = 0; i < 2; ++i) {
            TableauHalf &h = i == 0 ? xs : zs;
            PauliString p = h[q];
            p.get_xs().for_each_word(p.get_zs(), h.get_signs(), [](SimdWord& x, SimdWord& z, SimdWord& s) {
                s ^= z;
            });
        }
    }

    void do_H(size_t q) {
        for (size_t i = 0; i < 2; ++i) {
            TableauHalf &h = i == 0 ? xs : zs;
            PauliString p = h[q];
            p.get_xs().for_each_word(p.get_zs(), h.get_signs(), [](SimdWord& x, SimdWord& z, SimdWord& s) {
                std::swap(x, z);
                s ^= x & z;
            });
        }
    }

    void do_ZCX(size_t q1, size_t q2) {
        for (size_t i = 0; i < 2; ++i) {
            TableauHalf generator = i == 0 ? xs : zs;
            PauliString ps1 = generator[q1];
            PauliString ps2 = generator[q2];
            ps1.get_xs().for_each_word(ps1.get_zs(), ps2.get_xs(), ps2.get_zs(), generator.get_signs(), [](SimdWord& cx, SimdWord& cz, SimdWord& tx, SimdWord& tz, SimdWord s) {
                s ^= (cz ^ tx).andnot(cx & tz);
                cz ^= tz;
                tx ^= cx;
            });
        }
    }

    void do_H_YZ(size_t q) {
        for (size_t i = 0; i < 2 ; ++i) {
            TableauHalf generator = i == 0 ? xs : zs;
            PauliString ps = generator[q];
            ps.get_xs().for_each_word(ps.get_zs(), generator.get_signs(), [](SimdWord &x, SimdWord &z, SimdWord &s) {
            s ^= z.andnot(x);
            x ^= z;
            });
        }
    }

    void do_X_inverse(size_t q) {
        zs[q].get_sign() ^= 1;
    }

    void do_Y_inverse(size_t q) {
        xs[q].get_sign() ^= 1;
        zs[q].get_sign() ^= 1;
    }

    void do_Z_inverse(size_t q) {
        xs[q].get_sign() ^= 1;
    }

    void do_H_inverse(size_t q) {
        xs[q].swap_with(zs[q]);
    }
    
    void do_CNOT_inverse(size_t control, size_t target) {
        zs[target] *= zs[control];
        xs[control] *= xs[target];
    }

    size_t get_q() const{
        return num_qbits;
    }

    const TableauHalf& get_xs() const{
        return xs;
    }

    const TableauHalf& get_zs() const{
        return zs;
    }

    void do_transposed() {
        xs.get_x_table().do_transposed();
        xs.get_z_table().do_transposed();
        zs.get_x_table().do_transposed();
        zs.get_z_table().do_transposed();
    }

    void to_str() {
        std::cout << "+-";
        for (size_t k = 0; k < num_qbits; k++) {
            std::cout << 'x';
            std::cout << 'z';
            std::cout << '-';
        }
        std::cout << "\n|";
        for (size_t k = 0; k < num_qbits; k++) {
            std::cout << ' ';
            std::cout << "+-"[xs[k].get_sign()];
            std::cout << "+-"[zs[k].get_sign()];
        }
        for (size_t q = 0; q < num_qbits; q++) {
            std::cout << "\n|";
            for (size_t k = 0; k < num_qbits; k++) {
                std::cout << ' ';
                auto x = xs[k];
                auto z = zs[k];
                std::cout << "_XZY"[x.get_xs()[q] + 2 * x.get_zs()[q]];
                std::cout << "_XZY"[z.get_xs()[q] + 2 * z.get_zs()[q]];
            }
        }
        std::cout << "\n";
    }
};