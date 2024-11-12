/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef PARA_H
#define PARA_H

namespace mindquantum::algorithm::qaia::detail {

struct Para {
    int B;
    float xi;
    float delta;
    float dt;
    int n_iter;

    Para() : B(0), xi(0.0f), delta(0.0f), dt(0.0f), n_iter(0) {
    }

    Para(int B_, float xi_, float delta_, float dt_, int n_iter_)
        : B(B_), xi(xi_), delta(delta_), dt(dt_), n_iter(n_iter_) {
    }
};

}  // namespace mindquantum::algorithm::qaia::detail

#endif
