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
#include <map>
#include <set>
#include <string>

#include "core/utils.h"

namespace mindquantum {
template <typename T>
struct ParameterResolver {
    MST<T> data_;
    SS no_grad_parameters_;
    SS requires_grad_parameters_;

    ParameterResolver() {
    }
    ParameterResolver(const MST<T> &data, const SS &ngp, const SS &rgp)
        : data_(data), no_grad_parameters_(ngp), requires_grad_parameters_(rgp) {
    }
    ParameterResolver(const VT<std::string> &names, const VT<T> &coeffs, const VT<bool> &requires_grads) {
        for (Index i = 0; i < static_cast<Index>(names.size()); ++i) {
            data_[names[i]] = coeffs[i];
            if (requires_grads[i]) {
                requires_grad_parameters_.insert(names[i]);
            } else {
                no_grad_parameters_.insert(names[i]);
            }
        }
    }
    void SetData(const VT<T> &data, const VS &name) {
        for (size_t i = 0; i < data.size(); i++) {
            data_[name[i]] = data[i];
        }
    }
    void Times(T f) {
        for (auto &it : data_) {
            data_[it.first] = -it.second;
        }
    }
};

template <typename T>
T LinearCombine(const ParameterResolver<T> &pr_big, const ParameterResolver<T> &pr_small) {
    T res = 0;
    for (typename MST<T>::const_iterator i = pr_small.data_.begin(); i != pr_small.data_.end(); ++i) {
        res += (pr_big.data_.at(i->first) * i->second);
    }
    return res;
}
}  // namespace mindquantum
#endif  // MINDQUANTUM_PR_PARAMETER_RESOLVER_H_
