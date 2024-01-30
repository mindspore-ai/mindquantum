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
#ifndef SIMULATOR_STABILIZER_QUERY_ELEMENT_H_
#define SIMULATOR_STABILIZER_QUERY_ELEMENT_H_
#include <utility>

#include "simulator/stabilizer/stabilizer.h"

namespace mindquantum::stabilizer {
std::pair<size_t, size_t> DetermineClass(size_t i);
void EvoClass1(StabilizerTableau* stab, size_t idx);
void EvoClass2(StabilizerTableau* stab, size_t idx);
void EvoClass3(StabilizerTableau* stab, size_t idx);
void EvoClass4(StabilizerTableau* stab, size_t idx);
StabilizerTableau QueryDoubleQubitsCliffordElem(size_t idx);
StabilizerTableau QuerySingleQubitCliffordElem(size_t idx);
void Verification();
}  // namespace mindquantum::stabilizer
#endif
