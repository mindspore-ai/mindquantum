/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CHEM_TIMING_H
#define INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CHEM_TIMING_H

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mindquantum::sim::chem::detail {

class ChemTimer {
 public:
    using clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<double, std::milli>;

    static ChemTimer& getInstance() {
        static ChemTimer instance;
        return instance;
    }

    void startTimer(const std::string& stage) {
        auto now = clock::now();
        start_times_[stage] = now;
    }

    void endTimer(const std::string& stage) {
        auto end = clock::now();
        auto start_it = start_times_.find(stage);
        if (start_it != start_times_.end()) {
            duration elapsed = end - start_it->second;
            std::lock_guard<std::mutex> lock(mutex_);
            cumulative_times_[stage] += elapsed.count();
            call_counts_[stage]++;
            start_times_.erase(start_it);
        }
    }

    void printReport() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "\n=== MQChem ApplyUCCGate Timing Report ===" << std::endl;
        std::cout << std::setw(40) << "Stage" << std::setw(15) << "Total (ms)" << std::setw(15) << "Calls"
                  << std::setw(15) << "Avg (ms)" << std::endl;
        std::cout << std::string(85, '-') << std::endl;

        double total_time = 0.0;
        for (const auto& [stage, time] : cumulative_times_) {
            auto calls = call_counts_[stage];
            double avg = calls > 0 ? time / calls : 0.0;
            std::cout << std::setw(40) << stage << std::setw(15) << std::fixed << std::setprecision(3) << time
                      << std::setw(15) << calls << std::setw(15) << std::fixed << std::setprecision(3) << avg
                      << std::endl;
            total_time += time;
        }

        std::cout << std::string(85, '-') << std::endl;
        std::cout << std::setw(40) << "TOTAL" << std::setw(15) << std::fixed << std::setprecision(3) << total_time
                  << std::endl;
        std::cout << "=====================================" << std::endl;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        cumulative_times_.clear();
        call_counts_.clear();
        start_times_.clear();
    }

 private:
    ChemTimer() = default;

    std::unordered_map<std::string, double> cumulative_times_;
    std::unordered_map<std::string, size_t> call_counts_;
    std::unordered_map<std::string, clock::time_point> start_times_;
    std::mutex mutex_;
};

class ScopedTimer {
 public:
    explicit ScopedTimer(const std::string& stage) : stage_(stage) {
        ChemTimer::getInstance().startTimer(stage_);
    }

    ~ScopedTimer() {
        ChemTimer::getInstance().endTimer(stage_);
    }

 private:
    std::string stage_;
};

}  // namespace mindquantum::sim::chem::detail

#endif  // INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CHEM_TIMING_H
