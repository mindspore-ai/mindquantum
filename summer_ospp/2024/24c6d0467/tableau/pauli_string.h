#include<cstdint>
#include "simd/simd_array.h"
#include "simd/bit.h"

// 将generator里面的数组解析成pauliString'
// 有必要再加这个数据结构吗？
struct PauliString {
    // 记录相位 -1为负 +1为正
    size_t num_qubits;
    bit sign;
    SimdArray *ptr_xs, *ptr_zs;

public:
    PauliString(SimdArray* ptr_xs, SimdArray* ptr_zs, bit sign) : ptr_xs(ptr_xs), ptr_zs(ptr_zs), sign(sign) {
    };

    PauliString& operator*=(const PauliString& other) {
        inplace_right_mul_returning_log_i_scalar(other);
        assert((log_i & 1) == 0);
        sign ^= log_i & 2;
        return *this;
    }

    uint8_t inplace_right_mul_returning_log_i_scalar(const PauliString &rhs) noexcept {
        assert(num_qubits == rhs.num_qubits);
        // Accumulator registers for counting mod 4 in parallel across each bit position.
        SimdWord cnt1;
        SimdWord cnt2;
        ptr_xs->for_each_word(
            ptr_zs, rhs.ptr_zs, rhs.ptr_zs, [&cnt1, &cnt2](SimdWord *x1, SimdWord *z1, SimdWord *x2, SimdWord *z2) {
                // Update the left hand side Paulis.
                auto old_x1 = x1;
                auto old_z1 = z1;
                *x1 ^= *x2;
                *z1 ^= *z2;

                // At each bit position: accumulate anti-commutation (+i or -i) counts.
                auto x1z2 = *old_x1 & *z2;
                auto anti_commutes = (*x2 & *old_z1) ^ x1z2;
                cnt2 ^= (cnt1 ^ *x1 ^ *z1 ^ x1z2) & anti_commutes;
                cnt1 ^= anti_commutes;
            });
        // Combine final anti-commutation phase tally (mod 4).
        auto s = (uint8_t)cnt1.popcount();
        s ^= cnt2.popcount() << 1;
        s ^= (uint8_t)rhs.sign << 1;
        return s & 3;
    }

    void prepend_H(size_t q) {
        xs[q].swap_with(zs[q]);
        sign ^= ptr_xs[q] && ptr_zs[q];
    }

    void prepend_PHASE(size_t q) {
        sign ^= ptr_xs[q] && ptr_zs[q];
        ptr_zs[q] ^= ptr_xs[q];
    } 
};