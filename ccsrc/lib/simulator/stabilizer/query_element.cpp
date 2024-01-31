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
#include <cassert>
#include <stdexcept>

#include "simulator/stabilizer/stabilizer.h"
namespace mindquantum::stabilizer {
std::pair<size_t, size_t> DetermineClass(size_t i) {
    if (i < 576) {
        return {0, i};
    }
    if (i < 5760) {
        return {1, i - 576};
    }
    if (i < 10944) {
        return {2, i - 5760};
    }
    return {3, i - 10944};
}

void EvoClass1(StabilizerTableau* stab, size_t idx) {
    if (idx >= 576) {
        throw std::runtime_error(fmt::format("idx ({}) should be less than 576.", idx));
    }
    std::vector<size_t> num{2, 2, 3, 3, 4, 4};
    std::vector<size_t> idxs;
    for (auto i : num) {
        idxs.push_back(idx % i);
        idx /= i;
    }
    if (idxs[0] == 1) {
        stab->ApplyH(0);
    }
    if (idxs[1] == 1) {
        stab->ApplyH(1);
    }
    if (idxs[2] == 1) {
        stab->ApplyV(0);
    } else if (idxs[2] == 2) {
        stab->ApplyW(0);
    }
    if (idxs[3] == 1) {
        stab->ApplyV(1);
    } else if (idxs[3] == 2) {
        stab->ApplyW(1);
    }
    if (idxs[4] == 1) {
        stab->ApplyX(0);
    } else if (idxs[4] == 2) {
        stab->ApplyY(0);
    } else if (idxs[4] == 3) {
        stab->ApplyZ(0);
    }
    if (idxs[5] == 1) {
        stab->ApplyX(1);
    } else if (idxs[5] == 2) {
        stab->ApplyY(1);
    } else if (idxs[5] == 3) {
        stab->ApplyZ(1);
    }
}

void EvoClass2(StabilizerTableau* stab, size_t idx) {
    if (idx >= 5184) {
        throw std::runtime_error(fmt::format("idx ({}) should be less than 5184.", idx));
    }
    std::vector<size_t> num{2, 2, 3, 3, 3, 3, 4, 4};
    std::vector<size_t> idxs;
    for (auto i : num) {
        idxs.push_back(idx % i);
        idx /= i;
    }
    if (idxs[0] == 1) {
        stab->ApplyH(0);
    }
    if (idxs[1] == 1) {
        stab->ApplyH(1);
    }
    if (idxs[2] == 1) {
        stab->ApplyV(0);
    } else if (idxs[2] == 2) {
        stab->ApplyW(0);
    }
    if (idxs[3] == 1) {
        stab->ApplyV(1);
    } else if (idxs[3] == 2) {
        stab->ApplyW(1);
    }
    stab->ApplyCNOT(1, 0);
    if (idxs[4] == 1) {
        stab->ApplyV(0);
    } else if (idxs[4] == 2) {
        stab->ApplyW(0);
    }
    if (idxs[5] == 1) {
        stab->ApplyV(1);
    } else if (idxs[5] == 2) {
        stab->ApplyW(1);
    }
    if (idxs[6] == 1) {
        stab->ApplyX(0);
    } else if (idxs[6] == 2) {
        stab->ApplyY(0);
    } else if (idxs[6] == 3) {
        stab->ApplyZ(0);
    }
    if (idxs[7] == 1) {
        stab->ApplyX(1);
    } else if (idxs[7] == 2) {
        stab->ApplyY(1);
    } else if (idxs[7] == 3) {
        stab->ApplyZ(1);
    }
}

void EvoClass3(StabilizerTableau* stab, size_t idx) {
    if (idx >= 5184) {
        throw std::runtime_error(fmt::format("idx ({}) should be less than 5184.", idx));
    }
    std::vector<size_t> num{2, 2, 3, 3, 3, 3, 4, 4};
    std::vector<size_t> idxs;
    for (auto i : num) {
        idxs.push_back(idx % i);
        idx /= i;
    }
    if (idxs[0] == 1) {
        stab->ApplyH(0);
    }
    if (idxs[1] == 1) {
        stab->ApplyH(1);
    }
    if (idxs[2] == 1) {
        stab->ApplyV(0);
    } else if (idxs[2] == 2) {
        stab->ApplyW(0);
    }
    if (idxs[3] == 1) {
        stab->ApplyV(1);
    } else if (idxs[3] == 2) {
        stab->ApplyW(1);
    }
    stab->ApplyCNOT(1, 0);
    stab->ApplyCNOT(0, 1);
    if (idxs[4] == 1) {
        stab->ApplyV(0);
    } else if (idxs[4] == 2) {
        stab->ApplyW(0);
    }
    if (idxs[5] == 1) {
        stab->ApplyV(1);
    } else if (idxs[5] == 2) {
        stab->ApplyW(1);
    }
    if (idxs[6] == 1) {
        stab->ApplyX(0);
    } else if (idxs[6] == 2) {
        stab->ApplyY(0);
    } else if (idxs[6] == 3) {
        stab->ApplyZ(0);
    }
    if (idxs[7] == 1) {
        stab->ApplyX(1);
    } else if (idxs[7] == 2) {
        stab->ApplyY(1);
    } else if (idxs[7] == 3) {
        stab->ApplyZ(1);
    }
}

void EvoClass4(StabilizerTableau* stab, size_t idx) {
    if (idx >= 576) {
        throw std::runtime_error(fmt::format("idx ({}) should be less than 576.", idx));
    }
    std::vector<size_t> num{2, 2, 3, 3, 4, 4};
    std::vector<size_t> idxs;
    for (auto i : num) {
        idxs.push_back(idx % i);
        idx /= i;
    }
    if (idxs[0] == 1) {
        stab->ApplyH(0);
    }
    if (idxs[1] == 1) {
        stab->ApplyH(1);
    }
    if (idxs[2] == 1) {
        stab->ApplyV(0);
    } else if (idxs[2] == 2) {
        stab->ApplyW(0);
    }
    if (idxs[3] == 1) {
        stab->ApplyV(1);
    } else if (idxs[3] == 2) {
        stab->ApplyW(1);
    }
    stab->ApplyCNOT(1, 0);
    stab->ApplyCNOT(0, 1);
    stab->ApplyCNOT(1, 0);
    if (idxs[4] == 1) {
        stab->ApplyX(0);
    } else if (idxs[4] == 2) {
        stab->ApplyY(0);
    } else if (idxs[4] == 3) {
        stab->ApplyZ(0);
    }
    if (idxs[5] == 1) {
        stab->ApplyX(1);
    } else if (idxs[5] == 2) {
        stab->ApplyY(1);
    } else if (idxs[5] == 3) {
        stab->ApplyZ(1);
    }
}

StabilizerTableau QueryDoubleQubitsCliffordElem(size_t idx) {
    if (idx >= 11520) {
        throw std::runtime_error(fmt::format("idx ({}) should be less than 11520.", idx));
    }
    StabilizerTableau stab(2);
    auto [cls, i] = DetermineClass(idx);
    switch (cls) {
        case 0: {
            EvoClass1(&stab, i);
            break;
        }
        case 1: {
            EvoClass2(&stab, i);
            break;
        }
        case 2: {
            EvoClass3(&stab, i);
            break;
        }
        default: {
            EvoClass4(&stab, i);
            break;
        }
    }
    return stab;
}

StabilizerTableau QuerySingleQubitCliffordElem(size_t idx) {
    if (idx >= 24) {
        throw std::runtime_error(fmt::format("idx ({}) should be less than 24.", idx));
    }
    auto idx_a = idx / 4;
    auto idx_b = idx % 4;
    StabilizerTableau stab(1);
    switch (idx_a) {
        case 1: {
            stab.ApplyH(0);
            break;
        }
        case 2: {
            stab.ApplySGate(0);
            break;
        }
        case 3: {
            stab.ApplyH(0);
            stab.ApplySGate(0);
            break;
        }
        case 4: {
            stab.ApplySGate(0);
            stab.ApplyH(0);
            break;
        }
        case 5: {
            stab.ApplyH(0);
            stab.ApplySGate(0);
            stab.ApplyH(0);
            break;
        }
        default:
            break;
    }
    switch (idx_b) {
        case 1: {
            stab.ApplyX(0);
            break;
        }
        case 2: {
            stab.ApplyY(0);
            break;
        }
        case 3: {
            stab.ApplyZ(0);
            break;
        }
        default:
            break;
    }
    return stab;
}

void Verification() {
    for (size_t i = 0; i < 24; i++) {
        StabilizerTableau stab = QuerySingleQubitCliffordElem(i);
        StabilizerTableau other(1);
        other.ApplyCircuit(stab.Decompose());
        if (!(stab == other)) {
            throw std::runtime_error(fmt::format("{}\n{}\n{}", i, stab.TableauToString(), other.TableauToString()));
        }
    }
    for (size_t i = 0; i < 11520; i++) {
        StabilizerTableau stab = QueryDoubleQubitsCliffordElem(i);
        StabilizerTableau other(2);
        other.ApplyCircuit(stab.Decompose());
        if (!(stab == other)) {
            throw std::runtime_error(fmt::format("{}", i));
        }
    }
}
}  // namespace mindquantum::stabilizer
