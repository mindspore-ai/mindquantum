#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace mindquantum::sim::stabilizer {

#define REG_OPERATOR(op_eq, op)                                                                                        \
    void operator op_eq(const LongBits& other) {                                                                       \
        if (other.n_bits != this->n_bits) {                                                                            \
            throw std::runtime_error("n_bits not equal.");                                                             \
        }                                                                                                              \
        for (size_t i = 0; i < data.size(); i++) {                                                                     \
            data[i] op_eq other.data[i];                                                                               \
        }                                                                                                              \
    }                                                                                                                  \
    LongBits operator op(const LongBits& other) const {                                                                \
        auto out = *this;                                                                                              \
        out op_eq other;                                                                                               \
        return out;                                                                                                    \
    }

class LongBits {
    using ele_t = uint64_t;
    constexpr static auto ele_size = sizeof(ele_t) * 8;

 public:
    LongBits() = default;
    explicit LongBits(size_t n_bits) : n_bits(n_bits) {
        if (n_bits == 0) {
            throw std::runtime_error("n_bits cannot be zero.");
        }
        auto n_ele = n_bits / ele_size + ((n_bits % ele_size) != 0);
        data = std::vector<ele_t>(n_ele, 0);
    }
    REG_OPERATOR(^=, ^);
    REG_OPERATOR(&=, &);

    void SetBit(size_t poi, bool val) {
        if (poi > n_bits - 1) {
            throw std::runtime_error("poi out of range.");
        }
        size_t index_in = poi % ele_size;
        size_t mask = static_cast<uint64_t>(1) << index_in;
        ele_t& ele = data[poi / ele_size];
        ele = (ele & ~mask) | mask;
    }

    size_t GetBit(size_t poi) const {
        if (poi > n_bits - 1) {
            throw std::runtime_error("poi out of range.");
        }
        return (data[poi / ele_size] >> (poi % ele_size)) & 1;
    }

    std::string ToString() const {
        std::string out = "";
        for (size_t i = 0; i < n_bits; i++) {
            out += (GetBit(i) == 0 ? "0" : "1");
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    void InplaceFlip() {
        for (auto& ele : data) {
            ele = ~ele;
        }
    }
    LongBits Flip() {
        auto out = *this;
        out.InplaceFlip();
        return out;
    }
    // end not included.
    bool Any(size_t start, size_t end) {
        if (end <= start) {
            throw std::runtime_error("end can not less than start." + std::to_string(start) + " "
                                     + std::to_string(end));
        }
        for (size_t i = start; i < end; i++) {
            if (GetBit(i) == 1) {
                return true;
            }
        }
        return false;
    }
    bool Any(size_t start) {
        return Any(start, n_bits);
    }

 private:
    LongBits(size_t n_bits, const std::vector<ele_t>& data) : n_bits(n_bits), data(data) {
    }

 private:
    size_t n_bits = 1;
    std::vector<ele_t> data = {0};
};

#undef REG_OPERATOR

// -----------------------------------------------------------------------------

class StabilizerTableau {
 public:
    explicit StabilizerTableau(size_t n_qubits) : n_qubits(n_qubits) {
        phase = LongBits(2 * n_qubits);
        table = std::vector<LongBits>(2 * n_qubits, LongBits(n_qubits * 2));
        for (size_t i = 0; i < 2 * n_qubits; i++) {
            table[i].SetBit(i, 1);
        }
    }

    std::string TableauToString() {
        std::string out = "";
        for (size_t i = 0; i < 2 * n_qubits; ++i) {
            for (size_t j = 0; j < 2 * n_qubits; ++j) {
                out += (table[j].GetBit(i) == 0 ? "0 " : "1 ");
                if (j + 1 == n_qubits) {
                    out += "| ";
                }
                if (j + 1 == 2 * n_qubits) {
                    out += "| ";
                    out += phase.GetBit(i) == 0 ? "0\n" : "1\n";
                }
            }
            if (i + 1 == n_qubits) {
                for (size_t j = 0; j < 4 * n_qubits + 5; j++) {
                    out += "-";
                }
                out += "\n";
            }
        }
        return out;
    }

    std::string StabilizerToString() {
        std::string out = "destabilizer:\n";
        for (size_t i = 0; i < n_qubits * 2; i++) {
            out += phase.GetBit(i) == 0 ? "+" : "-";
            for (size_t j = 0; j < n_qubits; j++) {
                switch ((table[j].GetBit(i) << 1) + table[j + n_qubits].GetBit(i)) {
                    case 0:
                        out += "I";
                        break;
                    case 1:
                        out += "Z";
                        break;
                    case 2:
                        out += "X";
                        break;
                    default:
                        out += "Y";
                        break;
                }
            }
            if (i + 1 != n_qubits * 2) {
                out += "\n";
            }
            if (i + 1 == n_qubits) {
                out += "stabilizer:\n";
            }
        }
        return out;
    }

    void ApplyX(size_t idx) {
        if (idx + 1 > n_qubits) {
            throw std::runtime_error("qubit out of range.");
        }
        phase ^= table[idx + n_qubits];
    }
    void ApplyZ(size_t idx) {
        if (idx + 1 > n_qubits) {
            throw std::runtime_error("qubit out of range.");
        }
        phase ^= table[idx];
    }
    void ApplyS(size_t idx) {
        if (idx + 1 > n_qubits) {
            throw std::runtime_error("qubit out of range.");
        }
        phase ^= (table[idx] & table[idx + n_qubits]);
        table[idx + n_qubits] ^= table[idx];
    }
    void ApplySdag(size_t idx) {
        if (idx + 1 > n_qubits) {
            throw std::runtime_error("qubit out of range.");
        }
        auto tmp = table[idx] & table[idx + n_qubits];
        tmp ^= table[idx];
        phase ^= tmp;
        table[idx + n_qubits] ^= table[idx];
    }

    void ApplyH(size_t idx) {
        if (idx + 1 > n_qubits) {
            throw std::runtime_error("qubit out of range.");
        }
        phase ^= table[idx] & table[idx + n_qubits];
        std::iter_swap(table.begin() + idx, table.begin() + idx + n_qubits);
    }

    void ApplyCNOT(size_t obj, size_t ctrl) {
        if (obj + 1 > n_qubits) {
            throw std::runtime_error("qubit out of range.");
        }
        if (ctrl + 1 > n_qubits) {
            throw std::runtime_error("qubit out of range.");
        }
        auto one = LongBits(2 * n_qubits);
        one.InplaceFlip();
        phase ^= table[ctrl] & table[obj + n_qubits] & (table[obj] ^ table[ctrl + n_qubits] ^ one);
        table[obj] ^= table[ctrl];
        table[ctrl + n_qubits] ^= table[obj + n_qubits];
    }
    std::vector<std::string> Decompose() const {
        printf("run decompose\n");
        std::vector<std::string> out;
        auto cpy = *this;
        for (size_t i = 0; i < n_qubits; ++i) {
            auto flag_aii_true = table[i].GetBit(i);
            for (size_t j = i + 1; j < n_qubits; ++j) {
                if (flag_aii_true) {
                    break;
                }
                if (cpy.table[i].GetBit(j)) {
                    cpy.ApplyCNOT(i, j);
                    out.push_back("CNOT " + std::to_string(i) + ", " + std::to_string(j));
                    flag_aii_true = 1;
                }
            }
            for (size_t j = i; j < n_qubits; ++j) {
                if (flag_aii_true) {
                    break;
                }
                if (cpy.table[i].GetBit(j + n_qubits)) {
                    cpy.ApplyH(j);
                    out.push_back("H " + std::to_string(j));
                    if (j != i) {
                        cpy.ApplyCNOT(i, j);
                        out.push_back("CNOT " + std::to_string(i) + ", " + std::to_string(j));
                        flag_aii_true = 1;
                    }
                }
            }
            for (size_t j = i + 1; j < n_qubits; j++) {
                if (cpy.table[i].GetBit(j)) {
                    cpy.ApplyCNOT(j, i);
                    out.push_back("CNOT " + std::to_string(j) + ", " + std::to_string(i));
                }
            }
            if (cpy.table[i].Any(i + n_qubits)) {
                if (!cpy.table[i].GetBit(i + n_qubits)) {
                    cpy.ApplyS(i);
                    out.push_back("S " + std::to_string(i));
                }
                for (size_t j = i + 1; j < n_qubits; j++) {
                    if (cpy.table[i].GetBit(j + n_qubits)) {
                        cpy.ApplyCNOT(i, j);
                        out.push_back("CNOT " + std::to_string(i) + ", " + std::to_string(j));
                    }
                }
                cpy.ApplyS(i);
                out.push_back("S " + std::to_string(i));
            }

            // ######  step04: make D a lower triangular  ######
            if (i + 1 < n_qubits and cpy.table[i + n_qubits].Any(i + n_qubits + 1)) {
                for (size_t j = i + 1; j < n_qubits; j++) {
                    if (cpy.table[i + n_qubits].GetBit(j + n_qubits)) {
                        cpy.ApplyCNOT(i, j);
                        out.push_back("CNOT " + std::to_string(i) + ", " + std::to_string(j));
                    }
                }
            }

            // ######  step05: make C a lower triangular  ######
            if (cpy.table[i + n_qubits].Any(i + 1)) {
                cpy.ApplyH(i);
                out.push_back("H " + std::to_string(i));
                for (int j = i + 1; j < n_qubits; j++) {
                    if (cpy.table[i + n_qubits].GetBit(j)) {
                        cpy.ApplyCNOT(j, i);
                        out.push_back("CNOT " + std::to_string(j) + ", " + std::to_string(i));
                    }
                }
                if (cpy.table[i + n_qubits].GetBit(i + n_qubits)) {
                    cpy.ApplyS(i);
                    out.push_back("S " + std::to_string(i));
                }
                cpy.ApplyH(i);
                out.push_back("H " + std::to_string(i));
            }
        }
        for (size_t i = 0; i < n_qubits; ++i) {
            if (cpy.phase.GetBit(i)) {
                out.push_back("Z " + std::to_string(i));
            }
            if (cpy.phase.GetBit(i + n_qubits)) {
                out.push_back("X " + std::to_string(i));
            }
        }
        return out;
    }

 private:
    size_t n_qubits;
    std::vector<LongBits> table;
    LongBits phase;
};
}  // namespace mindquantum::sim::stabilizer

int main() {
    using namespace mindquantum::sim::stabilizer;
    auto stab = StabilizerTableau(3);
    stab.ApplyH(0);
    stab.ApplyCNOT(1, 0);
    std::cout << stab.TableauToString() << std::endl;
    std::cout << stab.StabilizerToString() << std::endl;
    for (auto s : stab.Decompose()) {
        std::cout << s << std::endl;
    }
}
