#ifndef THIRD_PARTY_PROJECTQ_THREADPOOL_H
#define THIRD_PARTY_PROJECTQ_THREADPOOL_H
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>
class Threadpool {
 private:
    std::vector<std::thread> threads;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> shutdown{false};
    std::atomic<int> idle{0};
    std::queue<std::function<void()>> tasks;

 public:
    size_t GetSize() {
        return threads.size();
    }
    size_t GetIdle() {
        return idle;
    }
    size_t IsTasksEmpty() {
        return tasks.empty();
    }
    inline static Threadpool &GetInstance() {
        static Threadpool instance;
        return instance;  // Get instance of threadpool
    }
    inline void AddThread(size_t n) {
        for (size_t i = 0; i < n; i++) {
            auto task = [this]() {
                while (true) {
                    std::function<void()> func;
                    {
                        std::unique_lock<std::mutex> lock(this->mtx);
                        auto sign = [this]() { return this->shutdown || !this->tasks.empty(); };
                        // wait
                        this->cv.wait(lock, sign);
                        // realse
                        if (this->shutdown && this->tasks.empty()) {
                            return;
                        }
                        this->idle++;
                        func = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    func();
                    this->idle--;
                }
            };
            threads.emplace_back(task);
        }
    }

    template <class T, class... Args>
    auto Push(T &&t, Args &&...args) -> std::future<typename std::result_of<T(Args...)>::type> {
        // receive task from tasks
        std::function<decltype(t(args...))()> recv_fun = std::bind(std::forward<T>(t), std::forward<Args>(args)...);
        auto ptr = std::make_shared<std::packaged_task<decltype(t(args...))()>>(recv_fun);
        // warp task
        std::function<void()> shell = [ptr]() { (*ptr)(); };
        tasks.emplace(shell);
        // notify one thread
        cv.notify_one();
        return ptr->get_future();
    }

    inline void ReSize(size_t n) {
        while (true) {
            if (this->idle == 0 && this->tasks.empty()) {
                break;
            }
        }
        this->shutdown = true;
        cv.notify_all();
        for (auto &thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        this->shutdown = false;
        // Clean All Threads and add new threads
        threads.resize(0);
        AddThread(n);
    }

    Threadpool() {
        this->shutdown = false;
        // Default numbers of thread is 1
        AddThread(1);
    }
    ~Threadpool() {
        // shutdown all threads
        std::unique_lock<std::mutex> lock(mtx);
        this->shutdown = true;
        cv.notify_all();
        for (std::thread &thread : threads)
            if (thread.joinable()) {
                thread.join();
            }
    }
};
#endif