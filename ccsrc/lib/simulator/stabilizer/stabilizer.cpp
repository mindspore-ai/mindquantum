/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "simulator/stabilizer/stabilizer.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

#include "core/mq_base_types.h"
#include "math/longbits/longbits.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"

namespace mindquantum::stabilizer {
StabilizerTableau::StabilizerTableau(size_t n_qubits, unsigned seed) : n_qubits(n_qubits), seed(seed), rnd_eng_(seed) {
    phase = LongBits(2 * n_qubits);
    table = std::vector<LongBits>(2 * n_qubits, LongBits(n_qubits * 2));
    for (size_t i = 0; i < 2 * n_qubits; i++) {
        table[i].SetBit(i, 1);
    }
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
}
StabilizerTableau::StabilizerTableau(size_t n_qubits, unsigned seed, const std::vector<LongBits>& table,
                                     const LongBits& phase)
    : n_qubits(n_qubits), seed(seed), table(table), phase(phase), rnd_eng_(seed) {
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
}
StabilizerTableau StabilizerTableau::copy() const {
    return StabilizerTableau(n_qubits, seed, table, phase);
}

void StabilizerTableau::Reset() {
    phase = LongBits(2 * n_qubits);
    table = std::vector<LongBits>(2 * n_qubits, LongBits(n_qubits * 2));
    for (size_t i = 0; i < 2 * n_qubits; i++) {
        table[i].SetBit(i, 1);
    }
}
void StabilizerTableau::SetSeed(unsigned new_seed) {
    this->seed = new_seed;
    this->rnd_eng_ = RndEngine(new_seed);
    std::uniform_real_distribution<double> dist(0., 1.);
    this->rng_ = std::bind(dist, std::ref(this->rnd_eng_));
}
void StabilizerTableau::AddQubit() {
    size_t old_n = n_qubits;
    n_qubits += 1;

    LongBits new_phase(2 * n_qubits);
    for (size_t i = 0; i < old_n; i++) {
        new_phase.SetBit(i, phase.GetBit(i));
        new_phase.SetBit(i + n_qubits, phase.GetBit(i + old_n));
    }
    phase = new_phase;

    for (auto& row : table) {
        LongBits new_row(2 * n_qubits);
        for (size_t i = 0; i < old_n; i++) {
            new_row.SetBit(i, row.GetBit(i));
            new_row.SetBit(i + n_qubits, row.GetBit(i + old_n));
        }
        row = new_row;
    }
    LongBits new_destabilizer(2 * n_qubits);
    new_destabilizer.SetBit(old_n, 1);
    table.insert(table.begin() + old_n, new_destabilizer);

    LongBits new_stabilizer(2 * n_qubits);
    new_stabilizer.SetBit(2 * old_n + 1, 1);
    table.insert(table.begin() + 2 * old_n + 1, new_stabilizer);
}
// -----------------------------------------------------------------------------

std::string StabilizerTableau::TableauToString() const {
    std::string out = "";
    for (size_t i = 0; i < 2 * n_qubits; ++i) {
        for (size_t j = 0; j < 2 * n_qubits; ++j) {
            out += (GetElement(i, j) == 0 ? "0 " : "1 ");
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

std::string StabilizerTableau::StabilizerToString() const {
    std::string out = "destabilizer:\n";
    for (size_t i = 0; i < n_qubits * 2; i++) {
        out += phase.GetBit(i) == 0 ? "+" : "-";
        for (int j = n_qubits - 1; j >= 0; --j) {
            switch ((GetElement(i, j) << 1) + GetElement(i, j + n_qubits)) {
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

size_t StabilizerTableau::GetElement(size_t row, size_t col) const {
    if (row >= 2 * n_qubits) {
        throw std::runtime_error(fmt::format("row should be less than {}, but get {}.", 2 * n_qubits, row));
    }
    if (col >= 2 * n_qubits) {
        throw std::runtime_error(fmt::format("col should be less than {}, but get {}.", 2 * n_qubits, row));
    }
    return table[col].GetBit(row);
}

bool StabilizerTableau::operator==(const StabilizerTableau& other) const {
    return n_qubits == other.n_qubits && phase == other.phase && table == other.table;
}

VVT<size_t> StabilizerTableau::TableauToVector() const {
    VVT<size_t> out(2 * n_qubits, VT<size_t>(2 * n_qubits + 1, 0));
    for (size_t i = 0; i < 2 * n_qubits; i++) {
        for (size_t j = 0; j < 2 * n_qubits; j++) {
            out[i][j] = GetElement(i, j);
        }
        out[i][2 * n_qubits] = phase.GetBit(i);
    }
    return out;
}

// -----------------------------------------------------------------------------

void StabilizerTableau::ApplyX(size_t idx) {
    if (idx + 1 > n_qubits) {
        throw std::runtime_error("qubit out of range.");
    }
    phase ^= table[idx + n_qubits];
}
void StabilizerTableau::ApplyY(size_t idx) {
    if (idx + 1 > n_qubits) {
        throw std::runtime_error("qubit out of range.");
    }
    phase ^= (table[idx + n_qubits] ^ table[idx]);
}
void StabilizerTableau::ApplyZ(size_t idx) {
    if (idx + 1 > n_qubits) {
        throw std::runtime_error("qubit out of range.");
    }
    phase ^= table[idx];
}
void StabilizerTableau::ApplySGate(size_t idx) {
    if (idx + 1 > n_qubits) {
        throw std::runtime_error("qubit out of range.");
    }
    phase ^= (table[idx] & table[idx + n_qubits]);
    table[idx + n_qubits] ^= table[idx];
}
void StabilizerTableau::ApplySdag(size_t idx) {
    if (idx + 1 > n_qubits) {
        throw std::runtime_error("qubit out of range.");
    }
    auto tmp = table[idx] & table[idx + n_qubits];
    tmp ^= table[idx];
    phase ^= tmp;
    table[idx + n_qubits] ^= table[idx];
}
void StabilizerTableau::ApplyV(size_t idx) {
    ApplySdag(idx);
    ApplyH(idx);
}
void StabilizerTableau::ApplyW(size_t idx) {
    ApplyH(idx);
    ApplySGate(idx);
}
void StabilizerTableau::ApplyH(size_t idx) {
    if (idx + 1 > n_qubits) {
        throw std::runtime_error("qubit out of range.");
    }
    phase ^= table[idx] & table[idx + n_qubits];
    std::iter_swap(table.begin() + idx, table.begin() + idx + n_qubits);
}
void StabilizerTableau::ApplyCNOT(size_t obj, size_t ctrl) {
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

int CalcG(size_t x1, size_t z1, size_t x2, size_t z2) {
    int xx1 = static_cast<int>(x1);
    int zz1 = static_cast<int>(z1);
    int xx2 = static_cast<int>(x2);
    int zz2 = static_cast<int>(z2);

    if (xx1 == 1 && zz1 == 1) {
        return zz2 - xx2;
    }
    if (xx1 == 0 && zz1 == 1) {
        return xx2 * (1 - 2 * zz2);
    }
    if (xx1 == 1 && zz1 == 0) {
        return zz2 * (2 * xx2 - 1);
    }
    return 0;
}

void StabilizerTableau::RowSum(size_t h, size_t i) {
    int r0 = 2 * (phase.GetBit(h) + phase.GetBit(i));
    for (size_t j = 0; j < n_qubits; ++j) {
        r0 += CalcG(GetElement(i, j), GetElement(i, j + n_qubits), GetElement(h, j), GetElement(h, j + n_qubits));
        table[j].SetBit(h, table[j].GetBit(h) ^ table[j].GetBit(i));
        table[j + n_qubits].SetBit(h, table[j + n_qubits].GetBit(h) ^ table[j + n_qubits].GetBit(i));
    }
    phase.SetBit(h, (((r0 % 4) + 4) % 4) / 2);
}

size_t StabilizerTableau::ApplyMeasurement(size_t a) {
    for (size_t p = n_qubits; p < 2 * n_qubits; ++p) {
        if (GetElement(p, a) == 1) {
            for (size_t i = 0; i < 2 * n_qubits; ++i) {
                if (i != p && GetElement(i, a) == 1) {
                    RowSum(i, p);
                }
            }
            for (size_t i = 0; i < 2 * n_qubits; ++i) {
                table[i].SetBit(p - n_qubits, table[i].GetBit(p));
                table[i].SetBit(p, 0);
            }
            phase.SetBit(p - n_qubits, phase.GetBit(p));
            table[a + n_qubits].SetBit(p, 1);
            size_t rp = (rng_() < 0.5) ? 0 : 1;
            phase.SetBit(p, rp);
            return rp;
        }
    }
    LongBits tail(2 * n_qubits + 1);
    for (size_t i = 0; i < n_qubits; i++) {
        if (GetElement(i, a) == 1) {
            int r0 = 2 * (tail.GetBit(2 * n_qubits) + phase.GetBit(i + n_qubits));
            for (size_t j = 0; j < n_qubits; ++j) {
                r0 += CalcG(GetElement(i + n_qubits, j), GetElement(i + n_qubits, j + n_qubits), tail.GetBit(j),
                            tail.GetBit(j + n_qubits));
                tail.SetBit(j, tail.GetBit(j) ^ table[j].GetBit(i + n_qubits));
                tail.SetBit(j + n_qubits, tail.GetBit(j + n_qubits) ^ table[j + n_qubits].GetBit(i + n_qubits));
            }
            tail.SetBit(2 * n_qubits, (((r0 % 4) + 4) % 4) / 2);
        }
    }
    return tail.GetBit(2 * n_qubits);
}

size_t StabilizerTableau::ApplyGate(GateID g_id, size_t obj, size_t ctrl) {
    switch (g_id) {
        case GateID::CNOT:
            ApplyCNOT(obj, ctrl);
            break;
        case GateID::X:
            ApplyX(obj);
            break;
        case GateID::Y:
            ApplyY(obj);
            break;
        case GateID::Z:
            ApplyZ(obj);
            break;
        case GateID::H:
            ApplyH(obj);
            break;
        case GateID::S:
            ApplySGate(obj);
            break;
        case GateID::Sdag:
            ApplySdag(obj);
            break;
        case GateID::I:
            break;
        case GateID::M:
            return ApplyMeasurement(obj);
        default:
            throw std::runtime_error(fmt::format("Non clifford gate {} is invalid for stabilizer simulator.", g_id));
    }
    return 2;
}
#define THROW_TOO_MUCH_CTRL(g_id, q_ids, limit)                                                                        \
    if (q_ids.size() > (limit)) {                                                                                      \
        throw std::runtime_error(fmt::format("Too much control qubit for simulate gate {} with stabilizer.", g_id));   \
    }

std::map<std::string, int> StabilizerTableau::ApplyCircuit(const stab_circ_t& circ) {
    std::map<std::string, int> result;
    for (auto& gate : circ) {
        auto g_id = gate->GetID();
        auto obj = gate->GetObjQubits();
        auto ctrl = gate->GetCtrlQubits();
        switch (g_id) {
            case GateID::X: {
                THROW_TOO_MUCH_CTRL(g_id, ctrl, 2)
                else if (ctrl.size() == 1) {  // NOLINT
                    ApplyCNOT(obj[0], ctrl[0]);
                }
                else {  // NOLINT
                    ApplyX(obj[0]);
                }
                break;
            }
            case GateID::CNOT: {
                THROW_TOO_MUCH_CTRL(g_id, ctrl, 1)
                ApplyCNOT(obj[0], obj[1]);
                break;
            }
            case GateID::Y: {
                THROW_TOO_MUCH_CTRL(g_id, ctrl, 1)
                ApplyY(obj[0]);
                break;
            }
            case GateID::Z: {
                THROW_TOO_MUCH_CTRL(g_id, ctrl, 1)
                ApplyZ(obj[0]);
                break;
            }
            case GateID::H: {
                THROW_TOO_MUCH_CTRL(g_id, ctrl, 1)
                ApplyH(obj[0]);
                break;
            }
            case GateID::S: {
                THROW_TOO_MUCH_CTRL(g_id, ctrl, 1)
                ApplySGate(obj[0]);
                break;
            }
            case GateID::Sdag: {
                THROW_TOO_MUCH_CTRL(g_id, ctrl, 1)
                ApplySdag(obj[0]);
                break;
            }
            case GateID::M: {
                result[static_cast<MeasureGate*>(gate.get())->Name()] = ApplyMeasurement(obj[0]);
            }
            case GateID::I:
                break;
            default:
                throw std::runtime_error(
                    fmt::format("Non clifford gate {} is invalid for stabilizer simulator.", g_id));
        }
    }
    return result;
}
#undef THROW_TOO_MUCH_CTRL
// -----------------------------------------------------------------------------

void AppendH(StabilizerTableau* stab, stab_circ_t* out, size_t obj) {
    stab->ApplyH(obj);
    if (!out->empty() && out->back()->GetID() == GateID::H && out->back()->GetObjQubits()[0] == obj
        && out->back()->GetCtrlQubits().size() == 0) {
        out->pop_back();
    } else {
        out->push_back(std::make_shared<HGate>(qbits_t{static_cast<qbit_t>(obj)}));
    }
}

stab_circ_t StabilizerTableau::Decompose() const {
    stab_circ_t out;
    auto cpy = *this;
    for (size_t i = 0; i < n_qubits; ++i) {
        // step01: make A full rank
        auto flag_aii_true = cpy.GetElement(i, i);
        for (size_t j = i + 1; j < n_qubits; ++j) {
            if (flag_aii_true) {
                break;
            }
            if (cpy.GetElement(i, j)) {
                cpy.ApplyCNOT(i, j);
                out.push_back(
                    std::make_shared<XGate>(qbits_t{static_cast<qbit_t>(i)}, qbits_t{static_cast<qbit_t>(j)}));
                flag_aii_true = 1;
            }
        }
        for (size_t j = i; j < n_qubits; ++j) {
            if (flag_aii_true) {
                break;
            }
            if (cpy.GetElement(i, j + n_qubits)) {
                AppendH(&cpy, &out, j);
                if (j != i) {
                    cpy.ApplyCNOT(i, j);
                    out.push_back(
                        std::make_shared<XGate>(qbits_t{static_cast<qbit_t>(i)}, qbits_t{static_cast<qbit_t>(j)}));
                }
                flag_aii_true = 1;
            }
        }
        // step02: make A a lower triangular
        for (size_t j = i + 1; j < n_qubits; j++) {
            if (cpy.GetElement(i, j)) {
                cpy.ApplyCNOT(j, i);
                out.push_back(
                    std::make_shared<XGate>(qbits_t{static_cast<qbit_t>(j)}, qbits_t{static_cast<qbit_t>(i)}));
            }
        }
        // step03: make B a lower triangular
        if (std::any_of(cpy.table.begin() + i + n_qubits, cpy.table.end(),
                        [&](const LongBits& b) { return b.GetBit(i) == 1; })) {
            if (!cpy.GetElement(i, i + n_qubits)) {
                cpy.ApplySGate(i);
                out.push_back(std::make_shared<SdagGate>(qbits_t{static_cast<qbit_t>(i)}));
            }
            for (size_t j = i + 1; j < n_qubits; j++) {
                if (cpy.GetElement(i, j + n_qubits)) {
                    cpy.ApplyCNOT(i, j);
                    out.push_back(
                        std::make_shared<XGate>(qbits_t{static_cast<qbit_t>(i)}, qbits_t{static_cast<qbit_t>(j)}));
                }
            }
            cpy.ApplySGate(i);
            out.push_back(std::make_shared<SdagGate>(qbits_t{static_cast<qbit_t>(i)}));
        }
        // step04: make D a lower triangular
        if (i + 1 < n_qubits
            && std::any_of(cpy.table.begin() + i + n_qubits + 1, cpy.table.end(),
                           [&](const LongBits& b) { return b.GetBit(i + n_qubits) == 1; })) {
            for (size_t j = i + 1; j < n_qubits; j++) {
                if (cpy.GetElement(i + n_qubits, j + n_qubits)) {
                    cpy.ApplyCNOT(i, j);
                    out.push_back(
                        std::make_shared<XGate>(qbits_t{static_cast<qbit_t>(i)}, qbits_t{static_cast<qbit_t>(j)}));
                }
            }
        }
        // step05: make C a lower triangular
        if (std::any_of(cpy.table.begin() + i + 1, cpy.table.end(),
                        [&](const LongBits& b) { return b.GetBit(i + n_qubits) == 1; })) {
            AppendH(&cpy, &out, i);
            for (int j = i + 1; j < n_qubits; j++) {
                if (cpy.GetElement(i + n_qubits, j)) {
                    cpy.ApplyCNOT(j, i);
                    out.push_back(
                        std::make_shared<XGate>(qbits_t{static_cast<qbit_t>(j)}, qbits_t{static_cast<qbit_t>(i)}));
                }
            }
            if (cpy.GetElement(i + n_qubits, i + n_qubits)) {
                cpy.ApplySGate(i);
                out.push_back(std::make_shared<SdagGate>(qbits_t{static_cast<qbit_t>(i)}));
            }
            AppendH(&cpy, &out, i);
        }
    }
    // step06: deal with phase
    for (size_t i = 0; i < n_qubits; ++i) {
        if (cpy.phase.GetBit(i)) {
            out.push_back(std::make_shared<ZGate>(qbits_t{static_cast<qbit_t>(i)}));
        }
        if (cpy.phase.GetBit(i + n_qubits)) {
            out.push_back(std::make_shared<XGate>(qbits_t{static_cast<qbit_t>(i)}));
        }
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// -----------------------------------------------------------------------------

VT<unsigned> StabilizerTableau::Sampling(const stab_circ_t& circ, size_t shots, const MST<size_t>& key_map,
                                         unsigned int seed) const {
    auto key_size = key_map.size();
    VT<unsigned> res(shots * key_size);
    RndEngine rnd_eng = RndEngine(seed);
    std::uniform_real_distribution<double> dist(1.0, (1 << 20) * 1.0);
    std::function<double()> rng = std::bind(dist, std::ref(rnd_eng));
    for (size_t i = 0; i < shots; i++) {
        StabilizerTableau sim = *this;
        sim.SetSeed(static_cast<unsigned>(rng()));
        auto res0 = sim.ApplyCircuit(circ);
        for (auto& [k, v] : res0) {
            res[i * key_size + key_map.at(k)] = v;
        }
    }
    return res;
}

VT<unsigned> StabilizerTableau::SamplingMeasurementEndingWithoutNoise(const stab_circ_t& circ, size_t shots,
                                                                      const MST<size_t>& key_map,
                                                                      unsigned int seed) const {
    RndEngine rnd_eng = RndEngine(seed);
    std::uniform_real_distribution<double> dist(1.0, (1 << 20) * 1.0);
    std::function<double()> rng = std::bind(dist, std::ref(rnd_eng));

    VT<int> already_measured(this->n_qubits, 0);
    VT<int> m_qids;
    stab_circ_t other_circ;
    stab_circ_t mea_circ;

    for (auto& g : circ) {
        if (g->GetID() == GateID::M) {
            auto m_qid = g->GetObjQubits()[0];
            if (already_measured[m_qid] != 0) {
                throw std::runtime_error("Quantum circuit is not a measurement ending circuit.");
            }
            already_measured[m_qid] = 1;
            m_qids.push_back(m_qid);
            mea_circ.push_back(g);
        } else {
            other_circ.push_back(g);
        }
    }
    StabilizerTableau sim = *this;
    sim.ApplyCircuit(other_circ);
    sim.SetSeed(static_cast<unsigned>(rng()));
    return sim.Sampling(mea_circ, shots, key_map, static_cast<unsigned>(rng()));
}
// -----------------------------------------------------------------------------

stab_circ_t CliffordCircDagger(const stab_circ_t& circ) {
    stab_circ_t out;
    for (const auto& gate : circ) {
        auto g_id = gate->GetID();
        switch (g_id) {
            case GateID::I:
            case GateID::X:
            case GateID::Y:
            case GateID::Z:
            case GateID::H:
            case GateID::CNOT:
                out.push_back(gate);
                break;
            case GateID::S:
                out.push_back(std::make_shared<SdagGate>(gate->GetObjQubits()));
                break;
            case GateID::Sdag:
                out.push_back(std::make_shared<SGate>(gate->GetObjQubits()));
                break;
            default:
                throw std::runtime_error(
                    fmt::format("Non clifford gate {} is invalid for stabilizer simulator.", g_id));
        }
    }
    std::reverse(out.begin(), out.end());
    return out;
}

bool StabilizerTableau::IsRandomMeasurement(size_t qubit) const {
    for (size_t p = n_qubits; p < 2 * n_qubits; ++p) {
        if (GetElement(p, qubit) == 1) {
            return true;
        }
    }
    return false;
}

double StabilizerTableau::GetExpectation(const VT<PauliTerm<double>>& ham_termlist) const {
    double expectation = 0.0;
    StabilizerTableau new_state = this->copy();
    new_state.AddQubit();
    for (const auto& term : ham_termlist) {
        if (term.first.empty()) {
            expectation += term.second;
            continue;
        }

        StabilizerTableau state = new_state.copy();
        size_t anc = state.n_qubits - 1;

        for (const auto& [qubit, pauli] : term.first) {
            switch (pauli) {
                case 'Z':
                    state.ApplyCNOT(anc, qubit);
                    break;
                case 'X':
                    state.ApplyH(qubit);
                    state.ApplyCNOT(anc, qubit);
                    state.ApplyH(qubit);
                    break;
                case 'Y':
                    state.ApplySdag(qubit);
                    state.ApplyH(anc);
                    state.ApplyCNOT(qubit, anc);
                    state.ApplySGate(qubit);
                    state.ApplyH(anc);
                    break;
                default:
                    throw std::runtime_error("Invalid Pauli operator.");
            }
        }

        if (!state.IsRandomMeasurement(anc)) {
            size_t result = state.ApplyMeasurement(anc);
            expectation += (result == 0 ? 1 : -1) * term.second;
        }
    }
    return expectation;
}
}  // namespace mindquantum::stabilizer
