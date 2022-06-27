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
#ifndef MINDQUANTUM_PR_PARAMETER_RESOLVER_H_
#define MINDQUANTUM_PR_PARAMETER_RESOLVER_H_

#include <algorithm>
#include <complex>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "core/utils.h"

namespace mindquantum {
template <typename T1, typename T2>
bool IsTwoNumberClose(T1 v1, T2 v2) {
    if (std::abs(v1 - v2) < PRECISION) {
        return true;
    }
    return false;
}

template <typename T>
std::set<T> operator-(const std::set<T>& s1, const std::set<T>& s2) {
    std::set<T> out;
    std::set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), std::inserter(out, out.begin()));
    return out;
}

template <typename T>
std::set<T> operator&(const std::set<T>& s1, const std::set<T>& s2) {
    std::set<T> out;
    std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), std::inserter(out, out.begin()));
    return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& s) {
    os << "(";
    for (ITER(p, s)) {
        os << *p << ", ";
    }
    os << ")";
    return os;
}

template <typename T>
T Conj(const T& value) {
    return value;
}

template <typename T>
std::complex<T> Conj(const std::complex<T>& value) {
    return std::conj(value);
}

template <typename T>
struct RemoveComplex {
    using type = T;
};

template <typename T>
struct RemoveComplex<std::complex<T>> {
    using type = T;
};

template <typename T>
struct ParameterResolver {
    MST<T> data_{};
    T const_value = 0;
    SS no_grad_parameters_{};
    SS encoder_parameters_{};
    ParameterResolver() {
    }
    explicit ParameterResolver(T const_value) : const_value(const_value) {
    }
    ParameterResolver(const MST<T>& data, T const_value) : data_(data), const_value(const_value) {
        for (ITER(p, this->data_)) {
            if (p->first == "") {
                throw std::runtime_error("Parameter name cannot be empty.");
            }
        }
    }
    ParameterResolver(const MST<T>& data, T const_value, const SS& ngp, const SS& ep)
        : data_(data), const_value(const_value), no_grad_parameters_(ngp), encoder_parameters_(ep) {
    }
    explicit ParameterResolver(const std::string& p) {
        this->data_[p] = 1;
    }
    size_t Size() const {
        return this->data_.size();
    }

    inline void SetConst(T const_value) {
        this->const_value = const_value;
    }

    inline auto NIndex(size_t n) const {
        if (n >= this->Size()) {
            throw std::runtime_error("ParameterResolver: Index out of range.");
        }
        auto index_p = this->data_.begin();
        std::advance(index_p, n);
        return index_p;
    }

    std::string GetKey(size_t index) const {
        return this->NIndex(index)->first;
    }

    T GetItem(const std::string& key) const {
        if (!this->Contains(key)) {
            throw std::runtime_error("parameter " + key + " not in this parameter resolver.");
        }
        return this->data_.at(key);
    }

    T GetItem(size_t index) const {
        return this->NIndex(index)->second;
    }

    inline void SetItem(const std::string& key, T value) {
        data_[key] = value;
    }

    inline void SetItems(const VS& name, const VT<T>& data) {
        if (name.size() != data.size()) {
            throw std::runtime_error("size of name and data mismatch.");
        }
        for (size_t i = 0; i < name.size(); i++) {
            this->SetItem(name[i], data[i]);
        }
    }

    bool IsConst() const {
        if (this->data_.size() == 0) {
            return true;
        }
        for (ITER(p, this->data_)) {
            if (!IsTwoNumberClose(p->second, 0.0)) {
                return false;
            }
        }
        return true;
    }

    bool IsNotZero() {
        if (!IsTwoNumberClose(this->const_value, 0.0)) {
            return true;
        }
        for (ITER(p, this->data_)) {
            if (!IsTwoNumberClose(p->second, 0.0)) {
                return true;
            }
        }
        return false;
    }

    inline bool Contains(const std::string& key) const {
        return this->data_.find(key) != this->data_.end();
    }

    inline bool NoGradContains(const std::string& key) const {
        return this->no_grad_parameters_.find(key) != this->no_grad_parameters_.end();
    }

    inline bool EncoderContains(const std::string& key) const {
        return this->encoder_parameters_.find(key) != this->encoder_parameters_.end();
    }

    inline SS GetAllParameters() const {
        SS all_params = {};
        for (ITER(p, this->data_)) {
            all_params.insert(p->first);
        }
        return all_params;
    }

    inline SS GetRequiresGradParameters() const {
        return this->GetAllParameters() - this->no_grad_parameters_;
    }

    inline SS GetAnsatzParameters() const {
        return this->GetAllParameters() - this->encoder_parameters_;
    }

    ParameterResolver<T>& operator+=(T value) {
        this->const_value += value;
        return *this;
    }

    ParameterResolver<T>& operator+=(const ParameterResolver<T>& other) {
        if ((this->encoder_parameters_.size() == 0) & (this->no_grad_parameters_.size() == 0)
            & (other.encoder_parameters_.size() == 0) & (other.no_grad_parameters_.size() == 0)) {
            for (ITER(p, other.data_)) {
                this->data_[p->first] += p->second;
            }
        } else {
            if (((this->encoder_parameters_ & other.GetAnsatzParameters()).size() != 0)
                | ((this->GetAnsatzParameters() & other.encoder_parameters_).size() != 0)) {
                throw std::runtime_error("encoder or ansatz property of parameter conflict.");
            }
            if (((this->no_grad_parameters_ & other.GetRequiresGradParameters()).size() != 0)
                | ((this->GetRequiresGradParameters() & other.no_grad_parameters_).size() != 0)) {
                throw std::runtime_error("gradient property of parameter conflict.");
            }

            for (ITER(p, other.data_)) {
                auto& key = p->first;
                auto& value = p->second;
                if (this->Contains(key)) {
                    this->data_[key] += value;
                } else {
                    this->SetItem(key, value);
                    if (other.EncoderContains(key)) {
                        this->encoder_parameters_.insert(key);
                    }
                    if (other.NoGradContains(key)) {
                        this->no_grad_parameters_.insert(key);
                    }
                }
            }
        }
        this->const_value += other.const_value;
        return *this;
    }

    const ParameterResolver<T> operator+(T value) const {
        auto pr = *this;
        pr += value;
        return pr;
    }

    const ParameterResolver<T> operator+(const ParameterResolver<T>& pr) const {
        auto out = *this;
        out += pr;
        return out;
    }

    const ParameterResolver<T> operator-() const {
        auto out = *this;
        out.const_value = -out.const_value;
        for (ITER(p, out.data_)) {
            out.data_[p->first] = -out.data_[p->first];
        }
        return out;
    }

    ParameterResolver<T>& operator-=(T value) {
        *this += (-value);
        return *this;
    }

    ParameterResolver<T>& operator-=(const ParameterResolver<T>& other) {
        auto tmp = other;
        *this += (-tmp);
        return *this;
    }

    const ParameterResolver<T> operator-(const ParameterResolver<T>& pr) const {
        auto out = pr;
        return *this + (-out);
    }

    const ParameterResolver<T> operator-(T value) const {
        auto out = *this;
        out -= value;
        return out;
    }

    ParameterResolver<T>& operator*=(T value) {
        this->const_value *= value;
        for (ITER(p, this->data_)) {
            this->data_[p->first] *= value;
        }
        return *this;
    }

    ParameterResolver<T>& operator*=(const ParameterResolver<T> other) {
        if (this->IsConst()) {
            for (ITER(p, other.data_)) {
                this->data_[p->first] = this->const_value * p->second;
                if (!this->Contains(p->first)) {
                    if (other.EncoderContains(p->first)) {
                        this->encoder_parameters_.insert(p->first);
                    }
                    if (other.NoGradContains(p->first)) {
                        this->no_grad_parameters_.insert(p->first);
                    }
                }
                this->const_value = 0;
            }
        } else {
            if (other.IsConst()) {
                for (ITER(p, this->data_)) {
                    this->data_[p->first] *= other.const_value;
                }
                this->const_value *= other.const_value;
            } else {
                throw std::runtime_error("Parameter resolver only support first order variable.");
            }
        }
        this->const_value *= other.const_value;
        return *this;
    }

    const ParameterResolver<T> operator*(T value) const {
        auto pr = *this;
        pr *= value;
        return pr;
    }

    const ParameterResolver<T> operator*(const ParameterResolver<T> other) const {
        auto pr = *this;
        pr *= other;
        return pr;
    }

    ParameterResolver<T>& operator/=(T value) {
        this->const_value /= value;
        for (ITER(p, this->data_)) {
            this->data_[p->first] /= value;
        }
        this->const_value /= value;
        return *this;
    }

    ParameterResolver<T>& operator/=(const ParameterResolver<T> other) {
        if (!other.IsConst()) {
            throw std::runtime_error("Cannot div a non constant ParameterResolver.");
        }
        for (ITER(p, this->data_)) {
            this->data_[p->first] /= other.const_value;
        }
        this->const_value /= other.const_value;
        return *this;
    }

    const ParameterResolver<T> operator/(T value) const {
        auto pr = *this;
        pr /= value;
        return pr;
    }

    const ParameterResolver<T> operator/(const ParameterResolver<T> other) const {
        auto out = *this;
        out /= other;
        return out;
    }

    void PrintInfo() const {
        std::cout << *this << std::endl;
    }

    std::string ToString() const {
        auto& pr = *this;
        std::ostringstream os;
        size_t i = 0;
        os << "{";
        for (ITER(p, pr.data_)) {
            os << "'" << p->first << "': " << p->second;
            if (i < pr.Size() - 1) {
                os << ", ";
            }
            i++;
        }
        os << "}, const: " << pr.const_value;
        return os.str();
    }

    ParameterResolver<T> Copy() {
        auto out = *this;
        return out;
    }

    bool operator==(T value) const {
        if (this->Size() != 0) {
            return false;
        }
        return IsTwoNumberClose(this->const_value, value);
    }

    bool operator==(const ParameterResolver<T> pr) const {
        if (!IsTwoNumberClose(this->const_value, pr.const_value)) {
            return false;
        }
        if (this->data_.size() != pr.data_.size()) {
            return false;
        }
        if (this->no_grad_parameters_.size() != pr.no_grad_parameters_.size()) {
            return false;
        }
        if (this->encoder_parameters_.size() != pr.encoder_parameters_.size()) {
            return false;
        }
        for (ITER(p, this->data_)) {
            if (!pr.Contains(p->first)) {
                return false;
            }
            if (!IsTwoNumberClose(p->second, pr.GetItem(p->first))) {
                return false;
            }
        }
        for (ITER(p, this->no_grad_parameters_)) {
            if (!pr.NoGradContains(*p)) {
                return false;
            }
        }
        for (ITER(p, this->encoder_parameters_)) {
            if (!pr.EncoderContains(*p)) {
                return false;
            }
        }
        return true;
    }

    std::vector<std::string> ParamsName() const {
        std::vector<std::string> pn;
        for (ITER(p, this->data_)) {
            pn.push_back(p->first);
        }
        return pn;
    }

    std::vector<T> ParaValue() const {
        std::vector<T> pv;
        for (ITER(p, this->data_)) {
            pv.push_back(p->second);
        }
        return pv;
    }

    void RequiresGrad() {
        this->no_grad_parameters_ = {};
    }

    void NoGrad() {
        this->no_grad_parameters_ = {};
        for (ITER(p, this->data_)) {
            this->no_grad_parameters_.insert(p->first);
        }
    }

    void RequiresGradPart(const std::vector<std::string>& names) {
        for (auto& name : names) {
            if (this->NoGradContains(name)) {
                this->no_grad_parameters_.erase(name);
            }
        }
    }

    void NoGradPart(const std::vector<std::string>& names) {
        for (auto& name : names) {
            if (this->Contains(name)) {
                this->no_grad_parameters_.insert(name);
            }
        }
    }

    void AnsatzPart(const std::vector<std::string>& names) {
        for (auto& name : names) {
            if (this->EncoderContains(name)) {
                this->encoder_parameters_.erase(name);
            }
        }
    }

    void EncoderPart(const std::vector<std::string>& names) {
        for (auto& name : names) {
            if (this->Contains(name)) {
                this->encoder_parameters_.insert(name);
            }
        }
    }

    void AsEncoder() {
        for (ITER(p, this->data_)) {
            this->encoder_parameters_.insert(p->first);
        }
    }

    void AsAnsatz() {
        this->encoder_parameters_ = {};
    }

    void Update(const ParameterResolver<T>& other) {
        if ((this->encoder_parameters_.size() == 0) & (this->no_grad_parameters_.size() == 0)
            & (other.encoder_parameters_.size() == 0) & (other.no_grad_parameters_.size() == 0)) {
            for (ITER(p, other.data_)) {
                this->data_[p->first] = p->second;
            }
        } else {
            if (((this->encoder_parameters_ & other.GetAnsatzParameters()).size() != 0)
                | ((this->GetAnsatzParameters() & other.encoder_parameters_).size() != 0)) {
                throw std::runtime_error("encoder or ansatz property of parameter conflict.");
            }
            if (((this->no_grad_parameters_ & other.GetRequiresGradParameters()).size() != 0)
                | ((this->GetRequiresGradParameters() & other.no_grad_parameters_).size() != 0)) {
                throw std::runtime_error("gradient property of parameter conflict.");
            }

            for (ITER(p, other.data_)) {
                auto& key = p->first;
                auto& value = p->second;
                if (this->Contains(key)) {
                    this->data_[key] = value;
                } else {
                    this->SetItem(key, value);
                    if (other.EncoderContains(key)) {
                        this->encoder_parameters_.insert(key);
                    }
                    if (other.NoGradContains(key)) {
                        this->no_grad_parameters_.insert(key);
                    }
                }
            }
        }
        this->const_value = other.const_value;
    }

    ParameterResolver<T> Conjugate() {
        auto out = *this;
        for (ITER(p, out.data_)) {
            out.data_[p->first] = Conj(p->second);
        }
        out.const_value = Conj(out.const_value);
        return out;
    }

    ParameterResolver<T> Combination(const ParameterResolver<T>& pr) const {
        auto c = this->const_value;
        for (ITER(p, this->data_)) {
            c += p->second * pr.GetItem(p->first);
        }
        return ParameterResolver<T>(c);
    }

    auto Real() const {
        using type = typename RemoveComplex<T>::type;
        ParameterResolver<type> pr = {};
        pr.const_value = std::real(this->const_value);
        for (ITER(p, this->data_)) {
            auto& key = p->first;
            pr.data_[p->first] = std::real(p->second);
            if (this->EncoderContains(key)) {
                pr.encoder_parameters_.insert(key);
            }
            if (this->NoGradContains(key)) {
                pr.no_grad_parameters_.insert(key);
            }
        }
        return pr;
    }

    auto Imag() const {
        using type = typename RemoveComplex<T>::type;
        ParameterResolver<type> pr = {};
        pr.const_value = std::imag(this->const_value);
        for (ITER(p, this->data_)) {
            auto& key = p->first;
            pr.data_[p->first] = std::imag(p->second);
            if (this->EncoderContains(key)) {
                pr.encoder_parameters_.insert(key);
            }
            if (this->NoGradContains(key)) {
                pr.no_grad_parameters_.insert(key);
            }
        }
        return pr;
    }

    T Pop(const std::string& key) {
        auto out = this->GetItem(key);
        this->data_.erase(key);
        if (this->EncoderContains(key)) {
            this->encoder_parameters_.erase(key);
        }
        if (this->NoGradContains(key)) {
            this->no_grad_parameters_.erase(key);
        }
        return out;
    }

    bool IsHermitian() {
        return *this == this->Conjugate();
    }

    bool IsAntiHermitian() {
        return *this == -this->Conjugate();
    }

    auto ToComplexPR() const {
        using type = typename RemoveComplex<T>::type;
        ParameterResolver<std::complex<type>> out;
        for (ITER(p, this->data_)) {
            auto& key = p->first;
            auto& t = p->second;
            out.data_[p->first] = std::complex<type>(t);
            if (this->EncoderContains(key)) {
                out.encoder_parameters_.insert(key);
            }
            if (this->NoGradContains(key)) {
                out.no_grad_parameters_.insert(key);
            }
        }
        out.const_value = std::complex<type>(this->const_value);
        return out;
    }

    bool IsComplexPR() const {
        using type = typename RemoveComplex<T>::type;
        return !std::is_same<type, T>::value;
    }
};

template <typename T>
ParameterResolver<T> operator+(T value, const ParameterResolver<T>& pr) {
    auto out = pr;
    out += value;
    return out;
}

template <typename T>
ParameterResolver<T> operator-(T value, const ParameterResolver<T>& pr) {
    auto out = pr;
    return (-out) + value;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const ParameterResolver<T>& pr) {
    os << pr.ToString();
    return os;
}

template <typename T>
ParameterResolver<T> operator*(T value, const ParameterResolver<T>& pr) {
    auto out = pr;
    out *= value;
    return out;
}

template <typename T>
ParameterResolver<T> operator/(T value, const ParameterResolver<T>& pr) {
    return ParameterResolver<T>(value) / pr;
}

template <typename T>
void BindPR(py::module* m, const std::string& name) {
    py::class_<ParameterResolver<T>, std::shared_ptr<ParameterResolver<T>>>(*m, name.c_str())
        .def(py::init<T>())
        .def(py::init<std::string>())
        .def(py::init<const MST<T>&, T>())
        .def(py::init<const MST<T>&, T, const SS&, const SS&>())
        .def_readonly("const", &ParameterResolver<T>::const_value)
        .def_readonly("data", &ParameterResolver<T>::data_)
        .def_readonly("no_grad_parameters", &ParameterResolver<T>::no_grad_parameters_)
        .def_readonly("encoder_parameters", &ParameterResolver<T>::encoder_parameters_)
        .def("set_const", &ParameterResolver<T>::SetConst)
        .def("params_name", &ParameterResolver<T>::ParamsName)
        .def("display", &ParameterResolver<T>::PrintInfo)
        .def("__setitem__", &ParameterResolver<T>::SetItem)
        .def("__getitem__", py::overload_cast<size_t>(&ParameterResolver<T>::GetItem, py::const_))
        .def("__getitem__", py::overload_cast<const std::string&>(&ParameterResolver<T>::GetItem, py::const_))
        .def("__len__", &ParameterResolver<T>::Size)
        .def("size", &ParameterResolver<T>::Size)
        .def("__bool__", &ParameterResolver<T>::IsNotZero)
        .def("__repr__", &ParameterResolver<T>::ToString)
        .def("__str__", &ParameterResolver<T>::ToString)
        .def("__contains__", &ParameterResolver<T>::Contains)
        .def("__copy__", &ParameterResolver<T>::Copy)
        .def("get_key", &ParameterResolver<T>::GetKey)
        .def(py::self + py::self)
        .def(T() + py::self)
        .def(py::self + T())
        .def(py::self - py::self)
        .def(T() - py::self)
        .def(py::self - T())
        .def(py::self * py::self)
        .def(py::self * T())
        .def(T() * py::self)
        .def(py::self / py::self)
        .def(T() / py::self)
        .def(py::self / T())
        .def(py::self == T())
        .def(py::self == py::self)
        .def(-py::self)
        .def("is_const", &ParameterResolver<T>::IsConst)
        .def("requires_grad", &ParameterResolver<T>::RequiresGrad)
        .def("no_grad", &ParameterResolver<T>::NoGrad)
        .def("no_grad_part", &ParameterResolver<T>::NoGradPart)
        .def("requires_grad_part", &ParameterResolver<T>::RequiresGradPart)
        .def("as_encoder", &ParameterResolver<T>::AsEncoder)
        .def("as_ansatz", &ParameterResolver<T>::AsAnsatz)
        .def("encoder_part", &ParameterResolver<T>::EncoderPart)
        .def("ansatz_part", &ParameterResolver<T>::AnsatzPart)
        .def("update", &ParameterResolver<T>::Update)
        .def("conjugate", &ParameterResolver<T>::Conjugate)
        .def("combination", &ParameterResolver<T>::Combination)
        .def("real", &ParameterResolver<T>::Real)
        .def("imag", &ParameterResolver<T>::Imag)
        .def("pop", &ParameterResolver<T>::Pop)
        .def("is_hermitian", &ParameterResolver<T>::IsHermitian)
        .def("is_anti_hermitian", &ParameterResolver<T>::IsAntiHermitian)
        .def("to_complex", &ParameterResolver<T>::ToComplexPR)
        .def("is_complex_pr", &ParameterResolver<T>::IsComplexPR);
}
}  // namespace mindquantum

#endif  // MINDQUANTUM_PR_PARAMETER_RESOLVER_H_
