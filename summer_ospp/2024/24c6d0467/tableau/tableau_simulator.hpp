#include "tableau/tableau.hpp"
#include "circuit/operate_unit.h"
#include "circuit/gate_type.h"
#include <random>

class TableauSimulator {

private:

    Tableau tableau;

    std::mt19937_64 rng;

    std::vector<bool> measure_result;

public:

    TableauSimulator(Tableau tableau, std::mt19937_64& rng) : tableau(tableau), rng(std::move(rng)) {}

    TableauSimulator(size_t num_qbits, std::mt19937_64 &rng) : tableau(Tableau(num_qbits)), rng(std::move(rng)) {}
        // 進行X門操作 
    void do_X(OperateUnit op) {
        tableau.do_X_inverse(op.qbit1);
    }
    // 進行Y門操作
    void do_Y(OperateUnit op) {
        tableau.do_Y_inverse(op.qbit1);
    }

    void do_Z(OperateUnit op) {
        tableau.do_Z_inverse(op.qbit1);
    }

    void do_H(OperateUnit op) {
        tableau.do_H_inverse(op.qbit1);
    }

    void do_CNOT(OperateUnit op) {
        assert(op.qbit2 != SIZE_MAX);
        tableau.do_CNOT_inverse(op.qbit1, op.qbit2);
    }

    //TODO 是否是确定态
    bool is_deterministic_x(size_t q) {
        return tableau.get_xs()[q].get_xs().not_zero();
    }

    size_t collapse_qubit_z(size_t q, Tableau& transposed_tableau) {
        // std::cout << "begin_collapse_qubit" << std::endl;
        // tableau.to_str();
        // transposed_tableau.to_str();
        auto n = tableau.get_q();
        size_t pivot = 0;
        while (pivot < n && !transposed_tableau.get_zs().get_x_table()[pivot][q]) {
            pivot++;
        }
        if (pivot == n) {
            return SIZE_MAX;
        }
        for (size_t k = pivot + 1; k < n; k++) {
            if (transposed_tableau.get_zs().get_x_table()[k][q]) {
                transposed_tableau.do_ZCX(pivot, k);
                // std::cout << "after_doZCX" << std::endl;
                // tableau.to_str();
            }
        }

        if (transposed_tableau.get_zs().get_z_table()[pivot][q]) {
            transposed_tableau.do_H_YZ(pivot);
            // std::cout << "after_do_H_YZ" << std::endl;
            // tableau.to_str();
        } else {
            transposed_tableau.do_H(pivot);
            // std::cout << "after_do_H" << std::endl;
            // tableau.to_str();
        }

        bool result_if_measured = rng() & 1;
        if (tableau.get_zs().get_signs()[q] != result_if_measured) {
            transposed_tableau.do_X(pivot);
            // std::cout << "after_do_X" << std::endl;
            // tableau.to_str();
        }
        // tableau.to_str();
        // transposed_tableau.to_str();
        // std::cout << "end_collapse_qubit" << std::endl;
        return pivot;
    }

    void do_MEASURE(OperateUnit op) {
        if(is_deterministic_x(op.qbit1)){
        } else {
            //TODO 优化
            // Tableau temp_transposed(tableau);
            // temp_transposed.do_transposed();
            tableau.do_transposed();
            collapse_qubit_z(op.qbit1, tableau);
            tableau.do_transposed();
        }
        // TODO 检测measure门是否结果正确
        bool b = tableau.get_zs().get_signs()[op.qbit1];
        measure_result.push_back(b);
    }

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
        case GateType::MEASURE :
            do_MEASURE(op);
        default:
            printf("Invalid Gate.\n");
            break;
        }
    }

    Tableau& get_tableau() {
        return tableau;
    }

    std::mt19937_64& get_rng() {
        return rng;
    }

    void print_measure_result() {
        std::cout << "measure_result" << std::endl;
        for (size_t i = 0; i < measure_result.size(); ++i) {
            std::cout << measure_result[i] << " ";
        }
        std::cout << std::endl;
    }
};