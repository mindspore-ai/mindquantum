#include<cstdint>

struct bit
{
    /* data */
    uint8_t *byte;

    uint16_t bit_index;

public:

    bit(void* base, uint16_t offset) : byte((uint8_t *)base + (offset / 8)), bit_index(offset & 7) {}

    bit operator^=(int value) {
        *byte ^= (uint8_t)value << bit_index;
    }

    operator bool() const {  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
        return (*byte >> bit_index) & 1;
    }
};
