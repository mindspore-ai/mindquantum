#include <stdint.h>
// 计算填充所需要的比特数
class utils{

public:
    static uint16_t popcount(uint16_t num) {
        int count = 0;
        while (num) {
            count += num & 1;
            num >>= 1;
        }
        return count;
    }
};