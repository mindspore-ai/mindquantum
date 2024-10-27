#pragma once
#include <cstdint>

class bit {

private:
    /* data */
    uint8_t *byte;

    uint16_t bit_index;

public:

    bit(void* base, uint16_t offset) : byte((uint8_t *)base + (offset / 8)), bit_index(offset & 7) {}

    bit operator^=(int value) {
        *byte ^= (uint8_t)value << bit_index;
        return *this;
    }

    operator bool() const {  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
        return (*byte >> bit_index) & 1;
    }

    bit &operator=(bool value) {
        *byte &= ~((uint8_t)1 << bit_index);
        *byte |= (uint8_t)value << bit_index;
        return *this;
    }

    bit operator=(const bit& value) {
        *this = (bool) value;
        return *this;
    }

    void swap_with(bit other) {
        bool b = (bool)other;
        other = (bool)*this;
        *this = b;
    }
};