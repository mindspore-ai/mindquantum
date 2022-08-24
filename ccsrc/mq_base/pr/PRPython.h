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

#ifndef PR_PRPYTHON_H_
#define PR_PRPYTHON_H_
#include <string>
#include <vector>

#include "pr/parameter_resolver.h"
namespace mindquantum {

enum class PRTypeID : uint8_t {
    PRTypeComplex128 = 1,
    PRTypeDouble = 2,
};

struct PRPython {
    using cdpr_t = ParameterResolver<std::complex<double>>;
    using dpr_t = ParameterResolver<double>;

    PRPython() = default;
    PRPython(const PRPython&) = default;
    PRPython(PRPython&&) noexcept = default;
    PRPython& operator=(const PRPython&) = default;
    PRPython& operator=(PRPython&&) noexcept = default;
    ~PRPython() noexcept = default;

    explicit PRPython(double value) : type_(PRTypeID::PRTypeDouble), pr_d_(dpr_t(value)) {
    }
    explicit PRPython(std::complex<double> value) : type_(PRTypeID::PRTypeComplex128), pr_cd_(cdpr_t(value)) {
    }
    explicit PRPython(const cdpr_t& pr) : type_(PRTypeID::PRTypeComplex128), pr_cd_(pr) {
    }
    explicit PRPython(const dpr_t& pr) : type_(PRTypeID::PRTypeDouble), pr_d_(pr) {
    }
    PRPython(const std::string& name, PRTypeID type) : type_(type) {
        if (type == PRTypeID::PRTypeComplex128) {
            pr_cd_ = cdpr_t(name);
        } else {
            pr_d_ = dpr_t(name);
        }
    }
    PRPython(const MST<double>& data, double const_value)
        : type_(PRTypeID::PRTypeDouble), pr_d_(dpr_t(data, const_value)) {
    }
    PRPython(const MST<std::complex<double>>& data, std::complex<double> const_value)
        : type_(PRTypeID::PRTypeComplex128), pr_cd_(cdpr_t(data, const_value)) {
    }
    inline void ConvertType(PRTypeID type) {
        if (type != type_) {
            if (type_ == PRTypeID::PRTypeDouble) {
                pr_cd_ = pr_d_.ToComplexPR();
                pr_d_ = dpr_t();
            } else {
                pr_d_ = pr_cd_.Real();
                pr_cd_ = cdpr_t();
            }
            type_ = type;
        }
    }
    PRPython& operator+=(double value) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_ += value;
        } else {
            pr_d_ += value;
        }
        return *this;
    }
    PRPython& operator+=(std::complex<double> value) {
        ConvertType(PRTypeID::PRTypeComplex128);
        pr_cd_ += value;
        return *this;
    }
    PRPython& operator+=(const dpr_t& other) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_ += other.ToComplexPR();
        } else {
            pr_d_ += other;
        }
        return *this;
    }
    PRPython& operator+=(const cdpr_t& other) {
        ConvertType(PRTypeID::PRTypeComplex128);
        pr_cd_ += other;
        return *this;
    }
    PRPython& operator+=(const PRPython& other) {
        if (other.IsComplexPR()) {
            *this += other.pr_cd_;
        } else {
            *this += other.pr_d_;
        }
        return *this;
    }
    PRPython& operator-=(double value) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_ -= value;
        } else {
            pr_d_ -= value;
        }
        return *this;
    }
    PRPython& operator-=(std::complex<double> value) {
        ConvertType(PRTypeID::PRTypeComplex128);
        pr_cd_ -= value;
        return *this;
    }
    PRPython& operator-=(const dpr_t& other) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_ -= other.ToComplexPR();
        } else {
            pr_d_ -= other;
        }
        return *this;
    }
    PRPython& operator-=(const cdpr_t& other) {
        ConvertType(PRTypeID::PRTypeComplex128);
        pr_cd_ -= other;
        return *this;
    }
    PRPython& operator-=(const PRPython& other) {
        if (other.IsComplexPR()) {
            *this -= other.pr_cd_;
        } else {
            *this -= other.pr_d_;
        }
        return *this;
    }
    PRPython& operator*=(double value) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_ *= value;
        } else {
            pr_d_ *= value;
        }
        return *this;
    }
    PRPython& operator*=(std::complex<double> value) {
        ConvertType(PRTypeID::PRTypeComplex128);
        pr_cd_ *= value;
        return *this;
    }
    PRPython& operator*=(const dpr_t& other) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_ *= other.ToComplexPR();
        } else {
            pr_d_ *= other;
        }
        return *this;
    }
    PRPython& operator*=(const cdpr_t& other) {
        ConvertType(PRTypeID::PRTypeComplex128);
        pr_cd_ *= other;
        return *this;
    }
    PRPython& operator*=(const PRPython& other) {
        if (other.IsComplexPR()) {
            *this *= other.pr_cd_;
        } else {
            *this *= other.pr_d_;
        }
        return *this;
    }
    PRPython& operator/=(double value) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_ /= value;
        } else {
            pr_d_ /= value;
        }
        return *this;
    }
    PRPython& operator/=(std::complex<double> value) {
        ConvertType(PRTypeID::PRTypeComplex128);
        pr_cd_ /= value;
        return *this;
    }
    PRPython& operator/=(const dpr_t& other) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_ /= other.ToComplexPR();
        } else {
            pr_d_ /= other;
        }
        return *this;
    }
    PRPython& operator/=(const cdpr_t& other) {
        ConvertType(PRTypeID::PRTypeComplex128);
        pr_cd_ += other;
        return *this;
    }
    PRPython& operator/=(const PRPython& other) {
        if (other.IsComplexPR()) {
            *this /= other.pr_cd_;
        } else {
            *this /= other.pr_d_;
        }
        return *this;
    }
    std::string ToString() const {
        if (type_ == PRTypeID::PRTypeComplex128) {
            return pr_cd_.ToString();
        }
        return pr_d_.ToString();
    }
    size_t Size() const {
        if (type_ == PRTypeID::PRTypeComplex128) {
            return pr_cd_.Size();
        }
        return pr_d_.Size();
    }
    inline void SetConst(double const_value) {
        if (type_ == PRTypeID::PRTypeDouble) {
            pr_d_.SetConst(const_value);
        } else {
            pr_cd_.SetConst(const_value);
        }
    }
    inline void SetConst(std::complex<double> const_value) {
        ConvertType(PRTypeID::PRTypeComplex128);
        pr_cd_.SetConst(const_value);
    }
    std::vector<std::string> ParamsName() const {
        if (type_ == PRTypeID::PRTypeComplex128) {
            return pr_cd_.ParamsName();
        }
        return pr_d_.ParamsName();
    }
    void PrintInfo() const {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_.PrintInfo();
        } else {
            pr_d_.PrintInfo();
        }
    }
    std::string GetKey(size_t index) const {
        if (type_ == PRTypeID::PRTypeComplex128) {
            return pr_cd_.GetKey(index);
        }
        return pr_d_.GetKey(index);
    }
    bool IsNotZero() const {
        if (type_ == PRTypeID::PRTypeComplex128) {
            return pr_cd_.IsNotZero();
        }
        return pr_d_.IsNotZero();
    }
    bool IsConst() const {
        if (type_ == PRTypeID::PRTypeComplex128) {
            return pr_cd_.IsConst();
        }
        return pr_d_.IsConst();
    }
    void RequiresGrad() {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_.RequiresGrad();
        } else {
            pr_d_.RequiresGrad();
        }
    }
    void NoGrad() {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_.NoGrad();
        } else {
            pr_d_.NoGrad();
        }
    }
    void NoGradPart(const std::vector<std::string>& names) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_.NoGradPart(names);
        } else {
            pr_d_.NoGradPart(names);
        }
    }
    void RequiresGradPart(const std::vector<std::string>& names) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_.RequiresGradPart(names);
        } else {
            pr_d_.RequiresGradPart(names);
        }
    }
    void AsEncoder() {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_.AsEncoder();
        } else {
            pr_d_.AsEncoder();
        }
    }
    void AsAnsatz() {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_.AsAnsatz();
        } else {
            pr_d_.AsAnsatz();
        }
    }
    void EncoderPart(const std::vector<std::string>& names) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_.EncoderPart(names);
        } else {
            pr_d_.EncoderPart(names);
        }
    }
    void AnsatzPart(const std::vector<std::string>& names) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_.AnsatzPart(names);
        } else {
            pr_d_.AnsatzPart(names);
        }
    }
    void Update(const cdpr_t& pr) {
        ConvertType(PRTypeID::PRTypeComplex128);
        pr_cd_.Update(pr);
    }
    void Update(const dpr_t& pr) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            pr_cd_.Update(pr.ToComplexPR());
        } else {
            pr_d_.Update(pr);
        }
    }
    void Update(const PRPython& pr) {
        if (pr.IsComplexPR()) {
            Update(pr.pr_cd_);
        } else {
            Update(pr.pr_d_);
        }
    }
    PRPython Conjugate() const {
        auto out = *this;
        if (type_ == PRTypeID::PRTypeComplex128) {
            out.pr_cd_ = pr_cd_.Conjugate();
        }
        return out;
    }
    PRPython Combination(const dpr_t& coeff) const {
        auto out = *this;
        if (type_ == PRTypeID::PRTypeComplex128) {
            out.pr_cd_ = out.pr_cd_.Combination(coeff.ToComplexPR());
        } else {
            out.pr_d_ = out.pr_d_.Combination(coeff);
        }
        return out;
    }
    PRPython Real() const {
        auto out = *this;
        if (type_ == PRTypeID::PRTypeComplex128) {
            out.pr_d_ = out.pr_cd_.Real();
            out.pr_cd_ = cdpr_t();
        }
        return out;
    }
    PRPython Imag() const {
        auto out = *this;
        if (type_ == PRTypeID::PRTypeComplex128) {
            out.pr_d_ = out.pr_cd_.Imag();
            out.pr_cd_ = cdpr_t();
        }
        return out;
    }
    bool IsHermitian() const {
        if (type_ == PRTypeID::PRTypeDouble) {
            return true;
        }
        return pr_cd_.IsHermitian();
    }
    bool IsAntiHermitian() const {
        if (type_ == PRTypeID::PRTypeDouble) {
            return true;
        }
        return pr_cd_.IsAntiHermitian();
    }
    PRPython ToComplexPR() const {
        auto out = *this;
        if (type_ == PRTypeID::PRTypeDouble) {
            out.pr_cd_ = out.pr_d_.ToComplexPR();
            out.pr_d_ = dpr_t();
        }
        return out;
    }
    bool IsComplexPR() const {
        return type_ == PRTypeID::PRTypeComplex128;
    }
    inline bool Contains(const std::string& name) {
        if (type_ == PRTypeID::PRTypeComplex128) {
            return pr_cd_.Contains(name);
        }
        return pr_d_.Contains(name);
    }
    PRTypeID type_ = PRTypeID::PRTypeComplex128;
    cdpr_t pr_cd_{};
    dpr_t pr_d_{};
};

}  // namespace mindquantum
#endif
