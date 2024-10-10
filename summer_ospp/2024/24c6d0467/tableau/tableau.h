#include<iostream>
#include "simd_word.h"
#include "simd_array.h"
#include "simd_table.h"
#include "bit.h"
class TableauHalf{
    SimdTable xt;
    SimdTable zt;
    SimdArray signs;
    //重構了[]運算符，
public:
    PauliString&& operator[](const uint16_t qbit) {
        return move(PauliString(xt[qbit], zt[qbit], signs.get_bit(qbit)));
    }
};

class Tableau {

    size_t num_qubits;

    TableauHalf xs;

    TableauHalf zs;
public:

    explicit Tableau(size_t num_qubits);

    bool operator==(const Tableau &other) const;

    bool operator!=(const Tableau &other) const;

    Tableau operator+(const Tableau &second) const;
    
    /// === Specialized vectorized methods for prepending operations onto the tableau === ///
    void prepend_X(size_t q) {
        zs[q].sign ^= 1;
    };

    void prepend_Y(size_t q) {
        xs[q].sign ^= 1;
        zs[q].sign ^= 1;
    };

    void prepend_Z(size_t q) {
        xs[q].sign ^= 1;
    };
    
    // TODO H门
    void prepend_H(size_t q) {
        for (int i = 0 ; i < n; i++) {
            xs[i].prepend_H(q);
            zs[i].prepend_H(q);
        }
    }

    void prepend_CNOT(size_t control, size_t target) {
        zs[target] *= zs[control];
        xs[control] *= xs[target];
    };

    //TODO 相位翻转门
    void prepend_PHASE(size_t q) {
        for (int i = 0 ; i < n; ++i) {
            xs[i].prepend_PHASE(q);
            zs[i].prepend_PHASE(q);
        }

    }
    
    // TODO:矩阵的逆
    void invesre() {

    }
};