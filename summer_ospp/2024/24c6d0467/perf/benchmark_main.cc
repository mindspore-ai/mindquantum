#include "perf/my_benchmark.hpp"
#include <vector>
#include <iostream>
std::vector<BenchmarkUnit>* all_benchmark = nullptr;
BenchmarkUnit* now_benchmark = nullptr;
int main() {
    if (all_benchmark == nullptr) {
        std::cout << "error:all_benchmark null ptr" << std::endl;
    }
    for (auto benchmark_unit : *all_benchmark) {
        now_benchmark = &benchmark_unit;
        benchmark_unit.do_func();
        benchmark_unit.print_result();
    }
    delete all_benchmark;
    return 0;
}