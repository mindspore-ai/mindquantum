#include "simd/simd_table.hpp"
#include "simd/simd_array.hpp"
#include "simd/simd_word.hpp"
#include "simd/bit.hpp"
#include <assert.h>

class PauliString {
    
    size_t num_qbits;

    SimdArray xs;

    SimdArray zs;

    bit sign;

public:

    PauliString(size_t num_qbits, size_t target, size_t minor_nums, SimdWord* ptr_xt, SimdWord* ptr_zt, SimdWord* ptr_signs) 
    : num_qbits(num_qbits), xs(SimdArray(ptr_xt + target * minor_nums, minor_nums)), zs(SimdArray(ptr_zt + target * minor_nums, minor_nums)), sign(bit(ptr_signs, target)) {}

    PauliString& operator*=(const PauliString& other) {
        uint8_t log_i = inplace_right_mul_returning_log_i_scalar(other);
        assert((log_i & 1) == 0);
        sign ^= log_i & 2;
        return *this;
    }

    uint8_t inplace_right_mul_returning_log_i_scalar(const PauliString &rhs) noexcept {
        assert(num_qbits == rhs.num_qbits);
        SimdWord cnt1{};
        SimdWord cnt2{};
        xs.for_each_word(
            zs, rhs.xs, rhs.zs, [&cnt1, &cnt2](SimdWord &x1, SimdWord &z1, SimdWord &x2, SimdWord &z2) {
                auto old_x1 = x1;
                auto old_z1 = z1;
                x1 ^= x2;
                z1 ^= z2;
                auto x1z2 = old_x1 & z2;
                auto anti_commutes = (x2 & old_z1) ^ x1z2;
                cnt2 ^= (cnt1 ^ x1 ^ z1 ^ x1z2) & anti_commutes;
                cnt1 ^= anti_commutes;
            });
        auto s = (uint8_t) cnt1.popcount();
        s ^= cnt2.popcount() << 1;
        s ^= (uint8_t)rhs.sign << 1;
        return s & 3;
    }
    
    void swap_with(PauliString other) {
        assert(num_qbits == other.num_qbits);
        sign.swap_with(other.sign);
        xs.swap_with(other.xs);
        zs.swap_with(other.zs);
    }

    bit& get_sign() {
        return sign;
    }

    SimdArray& get_xs() {
        return xs;
    }

    SimdArray& get_zs() {
        return zs;
    }
};