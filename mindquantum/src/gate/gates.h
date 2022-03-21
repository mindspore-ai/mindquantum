/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDQUANTUM_GATE_GATES_H_
#define MINDQUANTUM_GATE_GATES_H_
#include <cmath>

#include <string>

#include "core/utils.h"
#include "gate/basic_gate.h"
#include "matrix/two_dim_matrix.h"

namespace mindquantum {
template <typename T>
BasicGate<T> XGate = {false, gX, SELFHERMITIAN, Dim2Matrix<T>{{{{0, 0}, {1, 0}}, {{1, 0}, {0, 0}}}}};

template <typename T>
BasicGate<T> YGate = {false, gY, SELFHERMITIAN, Dim2Matrix<T>{{{{0, 0}, {0, -1}}, {{0, 1}, {0, 0}}}}};

template <typename T>
BasicGate<T> ZGate = {false, gZ, SELFHERMITIAN, Dim2Matrix<T>{{{{1, 0}, {0, 0}}, {{0, 0}, {-1, 0}}}}};

template <typename T>
BasicGate<T> IGate = {false, gI, SELFHERMITIAN, Dim2Matrix<T>{{{{1, 0}, {0, 0}}, {{0, 0}, {1, 0}}}}};

template <typename T>
BasicGate<T> HGate = {false, gH, SELFHERMITIAN,
                      Dim2Matrix<T>{{{{static_cast<T>(M_SQRT1_2), 0}, {static_cast<T>(M_SQRT1_2), 0}},
                                     {{static_cast<T>(M_SQRT1_2), 0}, {-static_cast<T>(M_SQRT1_2), 0}}}}};

template <typename T>
BasicGate<T> TGate = {
    false, gT, DOHERMITIAN,
    Dim2Matrix<T>{{{{1, 0}, {0, 0}}, {{0, 0}, {static_cast<T>(M_SQRT1_2), static_cast<T>(M_SQRT1_2)}}}}};

template <typename T>
BasicGate<T> SGate = {false, gS, DOHERMITIAN, Dim2Matrix<T>{{{{1, 0}, {0, 0}}, {{0, 0}, {0, 1}}}}};

template <typename T>
BasicGate<T> CNOTGate = {false, gCNOT, DOHERMITIAN,
                         Dim2Matrix<T>{{{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                        {{0, 0}, {1, 0}, {0, 0}, {0, 0}},
                                        {{0, 0}, {0, 0}, {0, 0}, {1, 0}},
                                        {{0, 0}, {0, 0}, {1, 0}, {0, 0}}}}};

template <typename T>
BasicGate<T> CZGate = {false, gCZ, SELFHERMITIAN,
                       Dim2Matrix<T>{{{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                      {{0, 0}, {1, 0}, {0, 0}, {0, 0}},
                                      {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
                                      {{0, 0}, {0, 0}, {0, 0}, {-1, 0}}}}};

template <typename T>
BasicGate<T> SWAPGate = {false, gSWAP, SELFHERMITIAN,
                         Dim2Matrix<T>{{{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                        {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
                                        {{0, 0}, {1, 0}, {0, 0}, {0, 0}},
                                        {{0, 0}, {0, 0}, {0, 0}, {1, 0}}}}};

template <typename T>
BasicGate<T> ISWAPGate = {false, gISWAP, DOHERMITIAN,
                          Dim2Matrix<T>{{{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                         {{0, 0}, {0, 0}, {0, 1}, {0, 0}},
                                         {{0, 0}, {0, 1}, {0, 0}, {0, 0}},
                                         {{0, 0}, {0, 0}, {0, 0}, {1, 0}}}}};

template <typename T>
BasicGate<T> RXGate = {
    true, gRX, PARAMSOPPOSITE,
    [](T theta) {
        return Dim2Matrix<T>{{{{COS1_2(theta), 0}, {0, -SIN1_2(theta)}}, {{0, -SIN1_2(theta)}, {COS1_2(theta), 0}}}};
    },
    [](T theta) {
        return Dim2Matrix<T>{
            {{{-SIN1_2(theta) / 2, 0}, {0, -COS1_2(theta) / 2}}, {{0, -COS1_2(theta) / 2}, {-SIN1_2(theta) / 2, 0}}}};
    }};

template <typename T>
BasicGate<T> RYGate = {
    true, gRY, PARAMSOPPOSITE,
    [](T theta) {
        return Dim2Matrix<T>{{{{COS1_2(theta), 0}, {-SIN1_2(theta), 0}}, {{SIN1_2(theta), 0}, {COS1_2(theta), 0}}}};
    },
    [](T theta) {
        return Dim2Matrix<T>{
            {{{-SIN1_2(theta) / 2, 0}, {-COS1_2(theta) / 2, 0}}, {{COS1_2(theta) / 2, 0}, {-SIN1_2(theta) / 2, 0}}}};
    }};

template <typename T>
BasicGate<T> RZGate = {
    true, gRZ, PARAMSOPPOSITE,
    [](T theta) {
        return Dim2Matrix<T>{{{{COS1_2(theta), -SIN1_2(theta)}, {0, 0}}, {{0, 0}, {COS1_2(theta), SIN1_2(theta)}}}};
    },
    [](T theta) {
        return Dim2Matrix<T>{
            {{{-SIN1_2(theta) / 2, -COS1_2(theta) / 2}, {0, 0}}, {{0, 0}, {-SIN1_2(theta) / 2, COS1_2(theta) / 2}}}};
    }};

template <typename T>
BasicGate<T> GPGate = {true, gGP, PARAMSOPPOSITE,
                       [](T theta) {
                           return Dim2Matrix<T>{{{{COS1_2(2 * theta), -SIN1_2(2 * theta)}, {0, 0}},
                                                 {{0, 0}, {COS1_2(2 * theta), -SIN1_2(2 * theta)}}}};
                       },
                       [](T theta) {
                           return Dim2Matrix<T>{{{{-SIN1_2(2 * theta), -COS1_2(2 * theta)}, {0, 0}},
                                                 {{0, 0}, {-SIN1_2(2 * theta), -COS1_2(2 * theta)}}}};
                       }};

template <typename T>
BasicGate<T> PSGate = {true, gPS, PARAMSOPPOSITE,
                       [](T theta) {
                           return Dim2Matrix<T>{{{{1, 0}, {0, 0}}, {{0, 0}, {COS1_2(2 * theta), SIN1_2(2 * theta)}}}};
                       },
                       [](T theta) {
                           return Dim2Matrix<T>{{{{0, 0}, {0, 0}}, {{0, 0}, {-SIN1_2(2 * theta), COS1_2(2 * theta)}}}};
                       }};

template <typename T>
BasicGate<T> XXGate = {true, gXX, PARAMSOPPOSITE,
                       [](T theta) {
                           return Dim2Matrix<T>{{{{COS1_2(2 * theta), 0}, {0, 0}, {0, 0}, {0, -SIN1_2(2 * theta)}},
                                                 {{0, 0}, {COS1_2(2 * theta), 0}, {0, -SIN1_2(2 * theta)}, {0, 0}},
                                                 {{0, 0}, {0, -SIN1_2(2 * theta)}, {COS1_2(2 * theta), 0}, {0, 0}},
                                                 {{0, -SIN1_2(2 * theta)}, {0, 0}, {0, 0}, {COS1_2(2 * theta), 0}}}};
                       },
                       [](T theta) {
                           return Dim2Matrix<T>{{{{-SIN1_2(2 * theta), 0}, {0, 0}, {0, 0}, {0, -COS1_2(2 * theta)}},
                                                 {{0, 0}, {-SIN1_2(2 * theta), 0}, {0, -COS1_2(2 * theta)}, {0, 0}},
                                                 {{0, 0}, {0, -COS1_2(2 * theta)}, {-SIN1_2(2 * theta), 0}, {0, 0}},
                                                 {{0, -COS1_2(2 * theta)}, {0, 0}, {0, 0}, {-SIN1_2(2 * theta), 0}}}};
                       }};

template <typename T>
BasicGate<T> YYGate = {true, gYY, PARAMSOPPOSITE,
                       [](T theta) {
                           return Dim2Matrix<T>{{{{COS1_2(2 * theta), 0}, {0, 0}, {0, 0}, {0, SIN1_2(2 * theta)}},
                                                 {{0, 0}, {COS1_2(2 * theta), 0}, {0, -SIN1_2(2 * theta)}, {0, 0}},
                                                 {{0, 0}, {0, -SIN1_2(2 * theta)}, {COS1_2(2 * theta), 0}, {0, 0}},
                                                 {{0, SIN1_2(2 * theta)}, {0, 0}, {0, 0}, {COS1_2(2 * theta), 0}}}};
                       },
                       [](T theta) {
                           return Dim2Matrix<T>{{{{-SIN1_2(2 * theta), 0}, {0, 0}, {0, 0}, {0, COS1_2(2 * theta)}},
                                                 {{0, 0}, {-SIN1_2(2 * theta), 0}, {0, -COS1_2(2 * theta)}, {0, 0}},
                                                 {{0, 0}, {0, -COS1_2(2 * theta)}, {-SIN1_2(2 * theta), 0}, {0, 0}},
                                                 {{0, COS1_2(2 * theta)}, {0, 0}, {0, 0}, {-SIN1_2(2 * theta), 0}}}};
                       }};

template <typename T>
BasicGate<T> ZZGate = {true, gZZ, PARAMSOPPOSITE,
                       [](T theta) {
                           return Dim2Matrix<T>{{{{COS1_2(2 * theta), -SIN1_2(2 * theta)}, {0, 0}, {0, 0}, {0, 0}},
                                                 {{0, 0}, {COS1_2(2 * theta), SIN1_2(2 * theta)}, {0, 0}, {0, 0}},
                                                 {{0, 0}, {0, 0}, {COS1_2(2 * theta), SIN1_2(2 * theta)}, {0, 0}},
                                                 {{0, 0}, {0, 0}, {0, 0}, {COS1_2(2 * theta), -SIN1_2(2 * theta)}}}};
                       },
                       [](T theta) {
                           return Dim2Matrix<T>{{{{-SIN1_2(2 * theta), -COS1_2(2 * theta)}, {0, 0}, {0, 0}, {0, 0}},
                                                 {{0, 0}, {-SIN1_2(2 * theta), COS1_2(2 * theta)}, {0, 0}, {0, 0}},
                                                 {{0, 0}, {0, 0}, {-SIN1_2(2 * theta), COS1_2(2 * theta)}, {0, 0}},
                                                 {{0, 0}, {0, 0}, {0, 0}, {-SIN1_2(2 * theta), -COS1_2(2 * theta)}}}};
                       }};

template <typename T>
BasicGate<T> GetMeasureGate(const std::string& name) {
    BasicGate<T> out;
    out.name_ = name;
    out.is_measure_ = true;
    return out;
}

template <typename T>
BasicGate<T> GetGateByName(const std::string& name) {
    BasicGate<T> out;
    if (name == gX) {
        out = XGate<T>;
    } else if (name == gY) {
        out = YGate<T>;
    } else if (name == gZ) {
        out = ZGate<T>;
    } else if (name == gI) {
        out = IGate<T>;
    } else if (name == gH) {
        out = HGate<T>;
    } else if (name == gT) {
        out = TGate<T>;
    } else if (name == gS) {
        out = SGate<T>;
    } else if (name == gCNOT) {
        out = CNOTGate<T>;
    } else if (name == gSWAP) {
        out = SWAPGate<T>;
    } else if (name == gISWAP) {
        out = ISWAPGate<T>;
    } else if (name == gCZ) {
        out = CZGate<T>;
    } else if (name == gRX) {
        out = RXGate<T>;
    } else if (name == gRY) {
        out = RYGate<T>;
    } else if (name == gRZ) {
        out = RZGate<T>;
    } else if (name == gPS) {
        out = PSGate<T>;
    } else if (name == gXX) {
        out = XXGate<T>;
    } else if (name == gYY) {
        out = YYGate<T>;
    } else if (name == gZZ) {
        out = ZZGate<T>;
        //    } else if (name == cPL) {
        //        out = PauliChannel<T>;
    } else if (name == gGP) {
        out = GPGate<T>;
    } else {
        auto msg = name + " not implement in backend!";
        throw std::invalid_argument(msg);
    }
    return out;
}
}  // namespace mindquantum
#endif  // MINDQUANTUM_GATE_GATES_H_
