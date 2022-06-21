//   Copyright 2021 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#ifndef PARAM_UTILS_HPP
#define PARAM_UTILS_HPP

#include <string>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>
#include <symengine/expression.h>

namespace Catch {
template <>
struct StringMaker<SymEngine::RCP<const SymEngine::Basic>> {
    static std::string convert(const SymEngine::RCP<const SymEngine::Basic>& a) {
        return str(*a);
    }
};
template <>
struct StringMaker<SymEngine::Expression> {
    static std::string convert(const SymEngine::Expression& e) {
        return str(e);
    }
};
}  // namespace Catch

struct Equals : Catch::MatcherBase<std::vector<SymEngine::RCP<const SymEngine::Basic>>> {
    template <typename... Ts>
    explicit Equals(Ts&&... ts) : comparator_{{std::forward<Ts>(ts)...}} {
    }

    bool match(const std::vector<SymEngine::RCP<const SymEngine::Basic>>& v) const override {
        if (comparator_.size() != v.size())
            return false;
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (!eq(*comparator_[i], *v[i])) {
                return false;
            }
        }
        return true;
    }
    std::string describe() const override {
        return "Equals: " + ::Catch::Detail::stringify(comparator_);
    }

    const std::vector<SymEngine::RCP<const SymEngine::Basic>> comparator_;
};

#endif /* PARAM_UTILS_HPP */
