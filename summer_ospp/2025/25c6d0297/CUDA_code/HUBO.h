#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>  // For dim3
#include <curand_kernel.h> // For curandState

using namespace std;

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << endl; \
        exit(1); \
    } \
}

// ==================== CUDA Kernel Prototypes ====================
// (Kernel implementations stay in HUBO.cu)

struct DeviceConfig;
DeviceConfig getDeviceConfig();

__global__ void initial_DSB(double* array, int batch_size, int num_spins, uint64_t seed);
__global__ void sign(double* array, int* sign_array, int batch_size, int num_spins);

template<typename T>
__global__ void calculate_poly_term(T* d_x_data, int* Spin_label, int* start_indices, int* end_indices,
    int batch_size, int num_spins, int num_groups,
    double* coeff_high_order, double* partial_energy);

template<typename T>
__global__ void calculate_gradient_term(T* d_x_data, int* Spin_group, int* start_indices, int* end_indices,
    int batch_size, int num_spins, int num_groups,
    double* partial_energy, double* coeff_one_order, double* gradient);

__global__ void update_BSB(double P, double xi, double* d_x, int batch_size, int num_spins,
    double dt, double delta, double* gradient_term, double* array_y);

__global__ void update_DSB(double P, double xi, double* d_x, int batch_size, int num_spins,
    double dt, double delta, double* gradient_term, double* array_y);

// ==================== Class Declarations ====================

struct DeviceConfig {
    int maxThreadsPerBlock;
    int maxSharedMemPerBlock;
    int warpSize;
    int optimalBlockSize;
    int maxBlocks;
};

class QAIA
{
protected:
    // Common parameters
    vector<vector<int>> J_high_order;        // High-order coupling relationships
    vector<int> J_one_order;                 // One-order coupling relationships
    vector<double> coeff_high_order;         // High-order coefficients
    vector<double> coeff_one_order;          // One-order coefficients
    int num_spins;                           // Number of spins
    int num_high_order_term;                 // Number of high-order terms
    int n_iter;                              // Number of iterations
    int batch_size;                          // Batch size
    string backend;                          // Computation backend
    vector<double> poly_term;                // Partial high-order terms
    vector<double> gradient_term;            // Spin gradient terms
    vector<double> energy_batch;             // Energy for each batch
    int opt_x;                               // Index of optimal batch

    // State variables
    vector<vector<double>> x;                // Spin values (batch_size x N)
    vector<vector<int>> x_sign;              // Signed spin values (batch_size x N)

    // GPU data structures
    vector<int> h_Spin_label, h_Spin_group;
    vector<int> h_Start_indices, h_End_indices;
    vector<int> h_Spin_label_gradient, h_Spin_group_gradient;
    vector<int> h_Start_indices_gradient, h_End_indices_gradient;
    int  num_Spin_label;

    // Thread configuration
    dim3 blockSize;
    dim3 gridSize_batch_spins;
    dim3 gridSize_batch_groups;

    // Device memory pointers
    double* d_x;
    int* d_x_sign;
    int* d_Spin_label;
    int* d_Spin_group_gradient;
    double* d_coeff_high_order_term;
    double* d_coeff_one_order_term;
    int* d_Start_indices;
    int* d_End_indices;
    int* d_Start_indices_gradient;
    int* d_End_indices_gradient;
    double* d_high_order_term_result;
    double* d_gradient_term;

public:
    QAIA(const vector<vector<int>>& J, const vector<double>& coeff, int iter, int b_size, const string& be);
    void caculate_energy();
    void sign_single(vector<double>& array_x, vector<int>& array_x_sign);
    void sign_batch(vector<vector<double>>& array_x, vector<vector<int>>& array_x_sign);
};

class BSB : public QAIA
{
protected:
    double xi;
    double dt;
    double delta;
    double P;
    vector<vector<double>> y;
    double* d_y;

public:
    BSB(const vector<vector<int>>& J, const vector<double>& coeff, int iter, int b_size, const string& be);
    void initial();
    void update();
    std::vector<double> get_solution() const;
    double get_energy() const;
};

class DSB : public QAIA
{
protected:
    double xi;
    double dt;
    double delta;
    double P;
    vector<vector<double>> y;
    double* d_y;

public:
    DSB(const vector<vector<int>>& J, const vector<double>& coeff, int iter, int b_size, const string& be);
    void initial();
    void update();
    std::vector<double> get_solution() const;
    double get_energy() const;
};