#include<iostream>
#include "tableau.h"
#include "circuit.h"
#include "circuit/gate_type.h"
#include "circuit/operate_unit.h"
#include "simd/simd_word.h"
class TableauSimulator{

    Tableau tableau;

    TableauSimulator(size_t num_qbits) : tableau(Tableau(num_qbits)) {};

public:
    // 進行X門操作
    void do_X(OperateUnit op) {
        tableau.prepend_X(op.qbit1);
    }
    // 進行Y門操作
    void do_Y(OperateUnit op) {
        tableau.prepend_Y(op.qbit1);
    }

    void do_Z(OperateUnit op) {
        tableau.prepend_Z(op.qbit1);
    }

    void do_H(OperateUnit op) {
        tableau.prepend_H(op.qbit1);
    }

    void do_PHASE(OperateUnit op) {
        tableau.prepend_PHASE(op.qbit1);
    }
    void do_CNOT(OperateUnit op) {
        tableau.prepend_CNOT(op.qbit1, op.qbit2);
    }


    

    // 执行各个gate的操作
    void doGate(OperateUnit op) {
        switch (op.gate)
        {
        case GateType::X :
            /* code */
            do_X(op);
            break;
        case GateType::Y :
            do_Y(op);
            break;
        case GateType::Z :
            do_Z(op);
            break;
        case GateType::CNOT :
            do_CNOT(op);
            break;
        case GateType::PHASE :
            do_PHASE(op);
            break;
        default:
            printf("Invalid Gate.\n");
            break;
        }
    }

    // TODO 后续改成stim库的模式 返回SimdWord 现在没有写measure 无法得到最后的结果 先测试CNOT X Y Z H P门的结果
    void doCircuit(Circuit circuit) {
        for (auto op : circuit.getOperationUnits()) {
            doGate(op);
        }
    }
};
