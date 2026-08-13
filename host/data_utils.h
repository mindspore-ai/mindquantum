#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include "acl/acl.h"

#define CHECK_ACL(x) \
    do { \
        aclError __ret = x; \
        if (__ret != ACL_ERROR_NONE) { \
            std::cerr << __FILE__ << ":" << __LINE__ << " " #x " failed, error code: " << __ret << std::endl; \
            return -1; \
        } \
    } while (0)

template<typename T>
void InitRandom(std::vector<T>& vec, T min, T max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);
    for (auto& v : vec) {
        v = dis(gen);
    }
}

// Added per instructions for specific initialization logic: X = 0.02 * (rand - 0.5)
template<typename T>
void InitRandomCentered(std::vector<T>& vec, T scale) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    for (auto& v : vec) {
        v = scale * (dis(gen) - 0.5);
    }
}

template<typename T>
void PrintSlice(const std::vector<T>& vec, size_t count) {
    size_t n = std::min(vec.size(), count);
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i] << " ";
    }
    std::cout << "..." << std::endl;
}

#endif