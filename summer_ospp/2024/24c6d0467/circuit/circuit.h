#include<iostream>
#include<vector>
#include<assert.h>
#include "circuit/gate_type.h"
#include "circuit/operate_unit.h"
using namespace std;

// 电路结构解析
struct Circuit{
    // TODO operateUnit结构优化 支持更多类型的量子门
    std::vector<OperateUnit> operations;
public:

    void addOperation(GateType gateType, size_t q) {
        assert(gateType == GateType::HADAMARD || gateType == GateType::Z || gateType == GateType::X || gateType == GateType::Y || gateType == GateType::PHASE);
        operations.push_back(OperateUnit(gateType, q));
    }

    void addOperation(GateType gateType, size_t q1, size_t q2) {
        assert(gateType == GateType::CNOT);
        operations.push_back(OperateUnit(gateType, q1, q2));
    }

    std::vector<OperateUnit> getOperationUnits() {
        return operations;
    }
};