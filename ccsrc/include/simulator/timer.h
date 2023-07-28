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

#ifndef INCLUDE_QUANTUMSTATE_TIMER_H_
#define INCLUDE_QUANTUMSTATE_TIMER_H_

#include <chrono>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mindquantum::timer {
using TimePoint = typename std::chrono::steady_clock::time_point;

TimePoint NOW();

template <typename time_unit_t>
int TimeDuration(TimePoint start, TimePoint end);

template <typename time_unit_t>
struct time_unit {
    static std::string_view time_unit_v();
};

template <>
struct time_unit<std::chrono::microseconds> {
    static std::string_view time_unit_v();
};

template <>
struct time_unit<std::chrono::milliseconds> {
    static std::string_view time_unit_v();
};

template <>
struct time_unit<std::chrono::seconds> {
    static std::string_view time_unit_v();
};

struct START {};
struct END {};

template <typename state_t_>
struct TimeState {
    TimeState() = default;
    explicit TimeState(TimePoint t) : time(t) {
    }
    TimePoint time;
};

using start_t = TimeState<START>;
using end_t = TimeState<END>;

struct TimePair {
    using time_t = std::pair<start_t, end_t>;
    using time_unit_t = std::chrono::microseconds;
    time_t time;
    bool finished = false;
    int dur = 0;
    explicit TimePair(TimePoint t) : time(std::make_pair(start_t(t), end_t())) {
    }
    TimePair(TimePoint start, TimePoint end)
        : time(std::make_pair(start_t(start), end_t(end))), finished(true), dur(TimeDuration<time_unit_t>(start, end)) {
    }
    void SetEnd(TimePoint end);
    bool Finished();
};

class Timer {
 public:
    void Start(const std::string &start);
    void End(const std::string &end);
    void EndAndStartOther(const std::string &end, const std::string &start);
    void Analyze();

 private:
    std::map<std::string, std::vector<TimePair>> data_{};
};
}  // namespace mindquantum::timer

#endif
