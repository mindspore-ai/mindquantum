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

#ifndef MINDQUANTUM_GATE_GATES_HPP_
#define MINDQUANTUM_GATE_GATES_HPP_

#include <cmath>

#include <functional>
#include <string>
#include <utility>

#include "core/parameter_resolver.hpp"
#include "core/two_dim_matrix.hpp"
#include "core/utils.hpp"
#include "ops/basic_gate.hpp"

#ifndef M_SQRT1_2
#    define M_SQRT1_2 0.707106781186547524400844362104849039
#endif  // !M_SQRT1_2

#ifndef M_PI
#    define M_PI 3.14159265358979323846264338327950288
#endif  // !M_PI

#ifndef M_PI_2
#    define M_PI_2 1.57079632679489661923132169163975144
#endif  // !M_PI_2

namespace mindquantum {
template <typename T>
BasicGate<T> XGate = {false, gX, SELFHERMITIAN, Dim2Matrix<T>{VVT<CT<T>>{{{0, 0}, {1, 0}}, {{1, 0}, {0, 0}}}}};

template <typename T>
BasicGate<T> YGate = {false, gY, SELFHERMITIAN, Dim2Matrix<T>{VVT<CT<T>>{{{0, 0}, {0, -1}}, {{0, 1}, {0, 0}}}}};

template <typename T>
BasicGate<T> ZGate = {false, gZ, SELFHERMITIAN, Dim2Matrix<T>{VVT<CT<T>>{{{1, 0}, {0, 0}}, {{0, 0}, {-1, 0}}}}};

template <typename T>
BasicGate<T> IGate = {false, gI, SELFHERMITIAN, Dim2Matrix<T>{VVT<CT<T>>{{{1, 0}, {0, 0}}, {{0, 0}, {1, 0}}}}};

template <typename T>
BasicGate<T> HGate = {false, gH, SELFHERMITIAN,
                      Dim2Matrix<T>{VVT<CT<T>>{{{static_cast<T>(M_SQRT1_2), 0}, {static_cast<T>(M_SQRT1_2), 0}},
                                               {{static_cast<T>(M_SQRT1_2), 0}, {-static_cast<T>(M_SQRT1_2), 0}}}}};

template <typename T>
BasicGate<T> TGate = {
    false, gT, DOHERMITIAN,
    Dim2Matrix<T>{{{{1, 0}, {0, 0}}, {{0, 0}, {static_cast<T>(M_SQRT1_2), static_cast<T>(M_SQRT1_2)}}}}};

template <typename T>
BasicGate<T> SGate = {false, gS, DOHERMITIAN, Dim2Matrix<T>{{{{1, 0}, {0, 0}}, {{0, 0}, {0, 1}}}}};

template <typename T>
BasicGate<T> CNOTGate = {false, gCNOT, DOHERMITIAN,
                         Dim2Matrix<T>{VVT<CT<T>>{{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                                  {{0, 0}, {1, 0}, {0, 0}, {0, 0}},
                                                  {{0, 0}, {0, 0}, {0, 0}, {1, 0}},
                                                  {{0, 0}, {0, 0}, {1, 0}, {0, 0}}}}};

template <typename T>
BasicGate<T> CZGate = {false, gCZ, SELFHERMITIAN,
                       Dim2Matrix<T>{VVT<CT<T>>{{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                                {{0, 0}, {1, 0}, {0, 0}, {0, 0}},
                                                {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
                                                {{0, 0}, {0, 0}, {0, 0}, {-1, 0}}}}};

template <typename T>
BasicGate<T> SWAPGate = {false, gSWAP, SELFHERMITIAN,
                         Dim2Matrix<T>{VVT<CT<T>>{{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                                  {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
                                                  {{0, 0}, {1, 0}, {0, 0}, {0, 0}},
                                                  {{0, 0}, {0, 0}, {0, 0}, {1, 0}}}}};

template <typename T>
BasicGate<T> ISWAPGate = {false, gISWAP, DOHERMITIAN,
                          Dim2Matrix<T>{VVT<CT<T>>{{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
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
        return Dim2Matrix<T>{VVT<CT<T>>{{{-SIN1_2(theta) / 2, 0}, {0, -COS1_2(theta) / 2}},
                                        {{0, -COS1_2(theta) / 2}, {-SIN1_2(theta) / 2, 0}}}};
    }};

template <typename T>
BasicGate<T> RYGate = {
    true, gRY, PARAMSOPPOSITE,
    [](T theta) {
        return Dim2Matrix<T>{{{{COS1_2(theta), 0}, {-SIN1_2(theta), 0}}, {{SIN1_2(theta), 0}, {COS1_2(theta), 0}}}};
    },
    [](T theta) {
        return Dim2Matrix<T>{VVT<CT<T>>{{{-SIN1_2(theta) / 2, 0}, {-COS1_2(theta) / 2, 0}},
                                        {{COS1_2(theta) / 2, 0}, {-SIN1_2(theta) / 2, 0}}}};
    }};

template <typename T>
BasicGate<T> RZGate = {
    true, gRZ, PARAMSOPPOSITE,
    [](T theta) {
        return Dim2Matrix<T>{{{{COS1_2(theta), -SIN1_2(theta)}, {0, 0}}, {{0, 0}, {COS1_2(theta), SIN1_2(theta)}}}};
    },
    [](T theta) {
        return Dim2Matrix<T>{VVT<CT<T>>{{{-SIN1_2(theta) / 2, -COS1_2(theta) / 2}, {0, 0}},
                                        {{0, 0}, {-SIN1_2(theta) / 2, COS1_2(theta) / 2}}}};
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
BasicGate<T> PSGate = {
    true, gPS, PARAMSOPPOSITE,
    [](T theta) {
        return Dim2Matrix<T>{VVT<CT<T>>{{{1, 0}, {0, 0}}, {{0, 0}, {COS1_2(2 * theta), SIN1_2(2 * theta)}}}};
    },
    [](T theta) {
        return Dim2Matrix<T>{VVT<CT<T>>{{{0, 0}, {0, 0}}, {{0, 0}, {-SIN1_2(2 * theta), COS1_2(2 * theta)}}}};
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
BasicGate<T> YYGate = {
    true, gYY, PARAMSOPPOSITE,
    [](T theta) {
        return Dim2Matrix<T>{VVT<CT<T>>{{{COS1_2(2 * theta), 0}, {0, 0}, {0, 0}, {0, SIN1_2(2 * theta)}},
                                        {{0, 0}, {COS1_2(2 * theta), 0}, {0, -SIN1_2(2 * theta)}, {0, 0}},
                                        {{0, 0}, {0, -SIN1_2(2 * theta)}, {COS1_2(2 * theta), 0}, {0, 0}},
                                        {{0, SIN1_2(2 * theta)}, {0, 0}, {0, 0}, {COS1_2(2 * theta), 0}}}};
    },
    [](T theta) {
        return Dim2Matrix<T>{VVT<CT<T>>{{{-SIN1_2(2 * theta), 0}, {0, 0}, {0, 0}, {0, COS1_2(2 * theta)}},
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
Dim2Matrix<T> U3Matrix(T theta, T phi, T lambda) {
    auto ct_2 = std::cos(theta / 2);
    auto st_2 = std::sin(theta / 2);
    auto el = std::exp(std::complex<T>(0, lambda));
    auto ep = std::exp(std::complex<T>(0, phi));
    auto elp = el * ep;
    return Dim2Matrix<T>({{ct_2, -el * st_2}, {ep * st_2, elp * ct_2}});
}

template <typename T>
Dim2Matrix<T> FSimMatrix(T theta, T phi) {
    auto a = std::cos(theta);
    auto b = CT<T>(0, -std::sin(theta));
    auto c = std::exp(std::complex<T>(0, phi));
    return Dim2Matrix<T>({{1, 0, 0, 0}, {0, a, b, 0}, {0, b, a, 0}, {0, 0, 0, c}});
}

template <typename T>
Dim2Matrix<T> U3DiffThetaMatrix(T theta, T phi, T lambda) {
    auto m = U3Matrix(theta + M_PI, phi, lambda);
    Dim2MatrixBinary<T>(&m, 0.5, std::multiplies<CT<T>>());
    return m;
}
template <typename T>
Dim2Matrix<T> FSimDiffThetaMatrix(T theta) {
    auto a = -std::sin(theta);
    auto b = CT<T>(0, -std::cos(theta));
    return Dim2Matrix<T>({{0, 0, 0, 0}, {0, a, b, 0}, {0, b, a, 0}, {0, 0, 0, 0}});
}

template <typename T>
Dim2Matrix<T> U3DiffPhiMatrix(T theta, T phi, T lambda) {
    auto m = U3Matrix(theta, phi + M_PI_2, lambda);
    m.matrix_[0][0] = 0;
    m.matrix_[0][1] = 0;
    return m;
}

template <typename T>
Dim2Matrix<T> FSimDiffPhiMatrix(T phi) {
    auto c = std::exp(std::complex<T>(0, phi + M_PI_2));
    return Dim2Matrix<T>({{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, c}});
}

template <typename T>
Dim2Matrix<T> U3DiffLambdaMatrix(T theta, T phi, T lambda) {
    auto m = U3Matrix(theta, phi, lambda + M_PI_2);
    m.matrix_[0][0] = 0;
    m.matrix_[1][0] = 0;
    return m;
}

template <typename T>
struct U3 : BasicGate<T> {
    ParameterResolver<T> theta;
    ParameterResolver<T> phi;
    ParameterResolver<T> lambda;
    std::pair<MST<size_t>, Dim2Matrix<T>> jacobi;
    VT<ParameterResolver<T>> prs;
    U3(const ParameterResolver<T>& theta, const ParameterResolver<T>& phi, const ParameterResolver<T>& lambda,
       const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits)
        : theta(theta), phi(phi), lambda(lambda) {
        this->name_ = "U3";
        this->parameterized_ = false;
        if (!this->theta.IsConst() || !this->phi.IsConst() || !this->lambda.IsConst()) {
            this->parameterized_ = true;
        }
        this->obj_qubits_ = obj_qubits;
        this->ctrl_qubits_ = ctrl_qubits;
        if (!this->parameterized_) {
            this->base_matrix_ = U3Matrix(theta.const_value, phi.const_value, lambda.const_value);
        }
        prs = {this->theta, this->phi, this->lambda};
        jacobi = Jacobi(prs);
    }
};

template <typename T>
struct FSim : BasicGate<T> {
    ParameterResolver<T> theta;
    ParameterResolver<T> phi;
    std::pair<MST<size_t>, Dim2Matrix<T>> jacobi;
    VT<ParameterResolver<T>> prs;
    FSim(const ParameterResolver<T>& theta, const ParameterResolver<T>& phi, const VT<Index>& obj_qubits,
         const VT<Index>& ctrl_qubits)
        : theta(theta), phi(phi) {
        this->name_ = "FSim";
        this->parameterized_ = false;
        if (!this->theta.IsConst() || !this->phi.IsConst()) {
            this->parameterized_ = true;
        }
        this->obj_qubits_ = obj_qubits;
        this->ctrl_qubits_ = ctrl_qubits;
        if (!this->parameterized_) {
            this->base_matrix_ = FSimMatrix(theta.const_value, phi.const_value);
        }
        prs = {this->theta, this->phi};
        jacobi = Jacobi(prs);
    }
};

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
#endif  // MINDQUANTUM_GATE_GATES_HPP_
