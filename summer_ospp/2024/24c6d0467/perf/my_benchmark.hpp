#include <chrono>
#include <functional>
#include <vector>
#include <string>
#include <iostream>

class BenchmarkResult {
private:
    size_t reps;
    size_t time;
    size_t memory;
public:

    size_t get_reps() const{
        return reps;
    }

    size_t get_memory() const{
        return memory;
    }

    size_t get_time() const{
        return time;
    }

    void set_reps(size_t reps) {
        this->reps = reps;
    }

    void set_time(size_t time) {
        this->time = time;
    }

    void set_memory(size_t memory) {
        this->memory = memory;
    }
};

class BenchmarkUnit {

private:

    std::string name;
    std::function<void(void)> func;
    BenchmarkResult result;

public:

    BenchmarkUnit(std::string name, std::function<void(void)> func) : name(name), func(func), result() {}

    void set_result(size_t reps, size_t time, size_t memory) {
        result.set_reps(reps);
        result.set_time(time);
        result.set_memory(memory);
    }

    void do_func() {
        func();
    }

    void print_result() {
        std::cout << "test_name:" << name << " reps:" << result.get_reps() << " cost_time:" << result.get_time() << "us" << std::endl;
    }
};

extern BenchmarkUnit* now_benchmark;
extern std::vector<BenchmarkUnit>* all_benchmark;

inline void add_benchmark(BenchmarkUnit benchmark) {
    if (all_benchmark == nullptr) {
        all_benchmark = new std::vector<BenchmarkUnit>();
    }
    all_benchmark->push_back(benchmark);
}

#define BENCHMARK(name, reps)           \
    void BENCHMARK_EXCUTE_##name##_METHOD();\
    struct BENCH_TYPE_##name {          \
        BENCH_TYPE_##name (){          \
            add_benchmark({#name, BENCHMARK_EXCUTE_##name##_METHOD});\
        }                               \
    }                                   \
    static BENCH_TYPE_##name## BENCH_TYPE_##name##_INSTANCE;\
    void BENCHMARK_EXCUTE_##name##_METHOD()

template <typename FUNC>
int64_t do_benchmark(FUNC body, size_t reps) {
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < reps; ++i) {
        body();
    }
    auto end = std::chrono::steady_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // TODO 增加内存计算模块
    now_benchmark->set_result(reps, micros, 0);
    return micros;
};