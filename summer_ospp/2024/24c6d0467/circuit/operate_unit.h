#include <cstddef>
#include <cstdint>
#include "circuit/gate_type.h"
#ifndef OPERATE_UNIT_H
#define OPERATE_UNIT_H

struct OperateUnit {

    GateType gate;

    size_t qbit1;

    size_t qbit2; // -1代表没有

public:
        
    OperateUnit(GateType gate, size_t qbit1, size_t qbit2) : gate(gate), qbit1(qbit1), qbit2(qbit2) {}

    OperateUnit(GateType gate, size_t qbit) : gate(gate), qbit1(qbit), qbit2(SIZE_MAX) {}
};
#endif