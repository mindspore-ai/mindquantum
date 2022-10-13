//   Copyright 2022 <Huawei Technologies Co., Ltd>
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

#include "simulator/timer.h"

#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
namespace mindquantum::timer {
TimePoint NOW() {
    return std::chrono::steady_clock::now();
}

template <typename time_unit_t>
int TimeDuration(TimePoint start, TimePoint end) {
    auto d = end - start;
    return std::chrono::duration_cast<time_unit_t>(d).count();
}

template <typename time_unit_t>
std::string_view time_unit<time_unit_t>::time_unit_v() {
    return "";
}

std::string_view time_unit<std::chrono::microseconds>::time_unit_v() {
    return "us";
}

std::string_view time_unit<std::chrono::milliseconds>::time_unit_v() {
    return "ms";
}

std::string_view time_unit<std::chrono::seconds>::time_unit_v() {
    return "s";
}

void TimePair::SetEnd(TimePoint end) {
    if (finished) {
        throw std::runtime_error("Already finished");
    }
    time.second = end_t(end);
    finished = true;
    dur = TimeDuration<time_unit_t>(time.first.time, time.second.time);
}

bool TimePair::Finished() {
    return finished;
}

void Timer::Start(const std::string &start) {
    if ((data_.find(start) != data_.end()) && (!data_[start].rbegin()->Finished())) {
        throw std::runtime_error("Previous timer not finished.");
    }
    data_[start].push_back(TimePair(NOW()));
}

void Timer::End(const std::string &end) {
    if (data_.find(end) == data_.end() || data_[end].rbegin()->Finished()) {
        throw std::runtime_error("Do not know to finish what.");
    }
    data_[end].rbegin()->SetEnd(NOW());
}

void Timer::EndAndStartOther(const std::string &end, const std::string &start) {
    Start(start);
    End(end);
}

void Timer::Analyze() {
    for (const auto &[name, time_data] : data_) {
        int this_time = std::accumulate(time_data.begin(), time_data.end(), 0.0,
                                        [&](int a, const TimePair &b) { return a + b.dur; });
        std::cout << "name: " << name << "\thit: " << time_data.size() << "\ttotal: " << this_time
                  << time_unit<TimePair::time_unit_t>::time_unit_v() << "\tpre: " << 1.0 * this_time / time_data.size()
                  << time_unit<TimePair::time_unit_t>::time_unit_v() << std::endl;
    }
}
}  // namespace mindquantum::timer
