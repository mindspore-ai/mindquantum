#include <iostream>
#include <fstream>
#include <vector>
#include "json.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono> 
#include <algorithm>
#include <climits>
#include <cmath>

using namespace std;
using json = nlohmann::json;
using namespace std::chrono;

#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << endl; \
        exit(1); \
    } \
}

struct DeviceConfig 
{
    int maxThreadsPerBlock;
    int maxSharedMemPerBlock;
    int warpSize;
    int optimalBlockSize;
    int maxBlocks;
};

DeviceConfig getDeviceConfig() 
{
    DeviceConfig config;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    config.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    config.maxSharedMemPerBlock = prop.sharedMemPerBlock;
    config.warpSize = prop.warpSize;

    config.optimalBlockSize = min(512, config.maxThreadsPerBlock);
    while (config.optimalBlockSize * sizeof(double) > config.maxSharedMemPerBlock / 2) 
    {
        config.optimalBlockSize /= 2;
    }
    config.optimalBlockSize = max(config.optimalBlockSize, config.warpSize);

    config.maxBlocks = prop.multiProcessorCount * 8;

    return config;
}

int count_diff(vector<int> a, vector<int> b, int size) 
{
    int count = 0;
    for (int i = 0; i < size; ++i) 
    {
        if (a[i] != b[i]) 
        {
            ++count;
        }
    }
    return count;
}

// CFC initialization kernel
__global__ void init_cfc_batch(double* array_x, double* array_e, int batch_size, int num_spins, uint64_t seed) 
{
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx < batch_size && spin_idx < num_spins) 
    {
        int global_idx = batch_idx * num_spins + spin_idx;
        curandState state;
        curand_init(seed, global_idx, 0, &state);
        float random = curand_uniform(&state);
        array_x[global_idx] = random * 2e-4 - 1e-4;
        array_e[global_idx] = 1.0;
    }
}

__global__ void calculate_poly_term_batch(double* d_x, int* Spin_label, int* start_indices, int* end_indices,
    int batch_size, int num_spins, int num_groups,
    double* coeff_high_order, double* partial_energy) 
{
    int batch_idx = blockIdx.x;
    int group_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx < batch_size && group_idx < num_groups) 
    {
        int start = start_indices[group_idx];
        int end = end_indices[group_idx];
        int result_idx = batch_idx * num_groups + group_idx;

        if (start != -1 && end != -1) 
        {
            double poly_term = 1.0;
            for (int i = start; i <= end; ++i) 
            {
                int spin_global_idx = batch_idx * num_spins + Spin_label[i];
                poly_term *= d_x[spin_global_idx];
            }
            poly_term *= coeff_high_order[group_idx];
            partial_energy[result_idx] = poly_term;
        }
        else 
        {
            partial_energy[result_idx] = 0.0;
        }
    }
}

__global__ void calculate_gradient_term_batch(double* d_x, int* Spin_group, int* start_indices, int* end_indices,
    int batch_size, int num_spins, int num_groups,
    double* partial_energy, double* coeff_one_order, double* gradient) 
{
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx < batch_size && spin_idx < num_spins) 
    {
        int start = start_indices[spin_idx];
        int end = end_indices[spin_idx];
        double grad = coeff_one_order[spin_idx];
        int gradient_idx = batch_idx * num_spins + spin_idx;
        int sign_idx = batch_idx * num_spins + spin_idx;

        if (start != -1 && end != -1 && d_x[sign_idx] != 0.0) 
        {
            for (int i = start; i <= end; ++i) 
            {
                int energy_idx = batch_idx * num_groups + Spin_group[i];
                grad += partial_energy[energy_idx] / d_x[sign_idx];
            }
        }
        gradient[gradient_idx] = grad;
    }
}

// CFC algorithm update kernel
__global__ void update_cfc_batch(double* d_x, double* d_e, double* gradient_term,
    double p, double xi, double dt, double alpha, double beta,
    int batch_size, int num_spins) 
{
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx < batch_size && spin_idx < num_spins) 
    {
        int global_idx = batch_idx * num_spins + spin_idx;
        double x = d_x[global_idx];
        double e = d_e[global_idx];
        double grad = gradient_term[global_idx];

        // CFC update equation: x = x + (-x^3 + (p-1)*x + xi*e*grad)*dt
        d_x[global_idx] = x + (-x * x * x + (p - 1.0) * x + xi * e * grad) * dt;

        // CFC special e update equation: e = e + (-beta * e * ((xi*e*grad)^2 - alpha)) * dt
        double field_term = xi * e * grad;
        double new_e = e + (-beta * e * (field_term * field_term - alpha)) * dt;

        // Ensure e doesn't become too small
        d_e[global_idx] = fmax(new_e, 0.01);
    }
}

// CFC constraint conditions
__global__ void apply_cfc_constraints(double* d_x, int batch_size, int num_spins) 
{
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx < batch_size && spin_idx < num_spins) 
    {
        int global_idx = batch_idx * num_spins + spin_idx;
        double limit = 1.5;

        if (abs(d_x[global_idx]) > limit) 
        {
            d_x[global_idx] = (d_x[global_idx] > 0) ? limit : -limit;
        }
    }
}

__global__ void sign_xy_batch(double* array, int batch_size, int num_spins, int* sign_array) 
{
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx < batch_size && spin_idx < num_spins) 
    {
        int global_idx = batch_idx * num_spins + spin_idx;
        sign_array[global_idx] = (array[global_idx] > 0) ? 1 : ((array[global_idx] < 0) ? -1 : 0);
    }
}

__global__ void calculate_energy_batch(int* d_x_sign, int* Spin_label, int* start_indices, int* end_indices,
    int batch_size, int num_spins, int num_groups,
    double* coeff_high_order, double* coeff_one_order, double* energies) 
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) 
    {
        double total_energy = 0.0;

        for (int g = 0; g < num_groups; ++g) 
        {
            int start = start_indices[g];
            int end = end_indices[g];

            if (start != -1 && end != -1) 
            {
                double poly_term = 1.0;
                for (int i = start; i <= end; ++i) 
                {
                    int spin_global_idx = batch_idx * num_spins + Spin_label[i];
                    poly_term *= d_x_sign[spin_global_idx];
                }
                total_energy += poly_term * coeff_high_order[g];
            }
        }

        for (int i = 0; i < num_spins; ++i) 
        {
            int spin_global_idx = batch_idx * num_spins + i;
            total_energy += d_x_sign[spin_global_idx] * coeff_one_order[i];
        }

        energies[batch_idx] = -total_energy;
    }
}

__global__ void find_min_energy_index(double* energies, int batch_size, int* min_index) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) 
    {
        double min_energy = energies[0];
        int min_idx = 0;

        for (int i = 1; i < batch_size; ++i) 
        {
            if (energies[i] < min_energy) 
            {
                min_energy = energies[i];
                min_idx = i;
            }
        }
        *min_index = min_idx;
    }
}

int main() 
{
    DeviceConfig config = getDeviceConfig();
    cout << "CFC Device Config - Max Threads: " << config.maxThreadsPerBlock
        << ", Optimal Block Size: " << config.optimalBlockSize << endl;

    int time_i = 0;
    const int batch_size = 100;

    while (true) 
    {
        //string filePath = "demo_data/" + to_string(time_i) + ".json";
        string filePath = "D:/HW/CUDAcode/DEMO/demo_data/" + to_string(time_i) + ".json";
        ifstream file(filePath);

        if (!file.is_open()) 
        {
            break;
        }
        cout << "DATA" << time_i << endl;

        json data;
        file >> data;
        file.close();

        vector<vector<int>> J = data["J"];
        vector<double> coeff = data["coeff"];


        vector<vector<int>> h_high_order_term;
        vector<int> h_one_order_term;

        for (const auto& row : J) 
        {
            if (row.size() > 1) 
            {
                h_high_order_term.push_back(row);
            }
            else {
                h_one_order_term.push_back(row[0]);
            }
        }

        int num_high_order_term = h_high_order_term.size();
        int num_spins = h_one_order_term.size();

        vector<double> h_coeff_high_order_term(coeff.begin(), coeff.begin() + num_high_order_term);
        vector<double> h_coeff_one_order_term(coeff.begin() + num_high_order_term, coeff.end());

        // CFC parameter calculation
        double sum_coeff_square = 0.0;
        for (double c : h_coeff_high_order_term) 
        {
            sum_coeff_square += c * c;
        }
        double xi = sqrt(2.0 * num_spins / sum_coeff_square);
        double dt = 0.075;
        int max_iters = 100;
        int Tr = static_cast<int>(0.9 * max_iters);
        double alpha = 1.0; // CFC alpha is constant
        double beta = 0.15;


        // Data preprocessing (same as BSB)
        vector<int> h_Spin_label;
        vector<int> h_Spin_group;
        for (int time_a = 0; time_a < h_high_order_term.size(); time_a++) 
        {
            for (int time_b = 0; time_b < h_high_order_term[time_a].size(); time_b++) 
            {
                h_Spin_label.push_back(h_high_order_term[time_a][time_b]);
                h_Spin_group.push_back(time_a);
            }
        }

        int num_Spin_label = h_Spin_label.size();
        vector<int> h_Start_indices(num_high_order_term, -1);
        vector<int> h_End_indices(num_high_order_term, -1);

        int current_group = -1;
        for (int time_c = 0; time_c < num_Spin_label; ++time_c) 
        {
            int group = h_Spin_group[time_c];
            if (group != current_group) 
            {
                if (current_group != -1) 
                {
                    h_End_indices[current_group] = time_c - 1;
                }
                h_Start_indices[group] = time_c;
                current_group = group;
            }
        }
        if (current_group != -1)
        {
            h_End_indices[current_group] = num_Spin_label - 1;
        }

        // Gradient index processing
        vector<int> h_Spin_label_gradient;
        vector<int> h_Spin_group_gradient;
        vector<int> h_Start_indices_gradient(num_spins, -1);
        vector<int> h_End_indices_gradient(num_spins, -1);

        vector<size_t> indices(h_Spin_label.size());
        for (size_t time_d = 0; time_d < indices.size(); ++time_d) 
        {
            indices[time_d] = time_d;
        }

        sort(indices.begin(), indices.end(), [&h_Spin_label](size_t time_e, size_t time_f) 
            {
            return h_Spin_label[time_e] < h_Spin_label[time_f];
            });

        for (size_t time_g : indices) 
        {
            h_Spin_label_gradient.push_back(h_Spin_label[time_g]);
            h_Spin_group_gradient.push_back(h_Spin_group[time_g]);
        }

        int current_group_1 = -1;
        for (int time_h = 0; time_h < num_Spin_label; ++time_h) 
        {
            int group_1 = h_Spin_label_gradient[time_h];
            if (group_1 != current_group_1) 
            {
                if (current_group_1 != -1) 
                {
                    h_End_indices_gradient[current_group_1] = time_h - 1;
                }
                h_Start_indices_gradient[group_1] = time_h;
                current_group_1 = group_1;
            }
        }
        if (current_group_1 != -1) 
        {
            h_End_indices_gradient[current_group_1] = num_Spin_label - 1;
        }

        // Thread configuration
        dim3 blockSize(256);
        dim3 gridSize_batch_spins(batch_size, (num_spins + blockSize.x - 1) / blockSize.x);
        dim3 gridSize_batch_groups(batch_size, (num_high_order_term + blockSize.x - 1) / blockSize.x);

        // Device memory allocation
        double* d_x = nullptr;
        double* d_e = nullptr; // CFC-specific field control variable
        int* d_x_sign = nullptr;
        double* d_high_order_term_result = nullptr;
        double* d_gradient_term = nullptr;
        double* d_energies = nullptr;
        int* d_min_index = nullptr;

        // Other device pointers
        int* d_Spin_label = nullptr;
        int* d_Spin_group_gradient = nullptr;
        double* d_coeff_high_order_term = nullptr;
        double* d_coeff_one_order_term = nullptr;
        int* d_Start_indices = nullptr;
        int* d_End_indices = nullptr;
        int* d_Start_indices_gradient = nullptr;
        int* d_End_indices_gradient = nullptr;

        CHECK_CUDA_ERROR(cudaMalloc(&d_x, batch_size * num_spins * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_e, batch_size * num_spins * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_x_sign, batch_size * num_spins * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_high_order_term_result, batch_size * num_high_order_term * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_gradient_term, batch_size * num_spins * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_energies, batch_size * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_min_index, sizeof(int)));

        CHECK_CUDA_ERROR(cudaMalloc(&d_Spin_label, num_Spin_label * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_Spin_group_gradient, num_Spin_label * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_coeff_high_order_term, num_high_order_term * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_coeff_one_order_term, num_spins * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_Start_indices, num_high_order_term * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_End_indices, num_high_order_term * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_Start_indices_gradient, num_spins * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_End_indices_gradient, num_spins * sizeof(int)));

        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_Spin_label, h_Spin_label.data(), num_Spin_label * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_Spin_group_gradient, h_Spin_group_gradient.data(), num_Spin_label * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_coeff_high_order_term, h_coeff_high_order_term.data(), num_high_order_term * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_coeff_one_order_term, h_coeff_one_order_term.data(), num_spins * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_Start_indices, h_Start_indices.data(), num_high_order_term * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_End_indices, h_End_indices.data(), num_high_order_term * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_Start_indices_gradient, h_Start_indices_gradient.data(), num_spins * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_End_indices_gradient, h_End_indices_gradient.data(), num_spins * sizeof(int), cudaMemcpyHostToDevice));

        // CFC initialization
        init_cfc_batch << <gridSize_batch_spins, blockSize >> > (d_x, d_e, batch_size, num_spins, time(NULL));

        auto start_time = high_resolution_clock::now();

        // CFC main loop
        for (int iters = 0; iters < max_iters; ++iters) 
        {
            // Calculate p value (alpha is constant in CFC)
            double p;
            if (iters < Tr) 
            {
                p = -0.5 + 1.5 * iters / (Tr - 1);
            }
            else {
                p = 1.0;
            }

            calculate_poly_term_batch << <gridSize_batch_groups, blockSize >> > (d_x, d_Spin_label, d_Start_indices, d_End_indices,
                batch_size, num_spins, num_high_order_term,
                d_coeff_high_order_term, d_high_order_term_result);

            calculate_gradient_term_batch << <gridSize_batch_spins, blockSize >> > (d_x, d_Spin_group_gradient,
                d_Start_indices_gradient, d_End_indices_gradient,
                batch_size, num_spins, num_high_order_term,
                d_high_order_term_result, d_coeff_one_order_term, d_gradient_term);

            update_cfc_batch << <gridSize_batch_spins, blockSize >> > (d_x, d_e, d_gradient_term,
                p, xi, dt, alpha, beta, batch_size, num_spins);

            apply_cfc_constraints << <gridSize_batch_spins, blockSize >> > (d_x, batch_size, num_spins);
        }

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        cout << "CFC total execution time: " << duration.count() / 1000.0 << " ms" << endl;

        // Calculate final energy and find optimal solution
        sign_xy_batch << <gridSize_batch_spins, blockSize >> > (d_x, batch_size, num_spins, d_x_sign);

        dim3 energyGrid((batch_size + 255) / 256);
        dim3 energyBlock(256);
        calculate_energy_batch << <energyGrid, energyBlock >> > (d_x_sign, d_Spin_label, d_Start_indices, d_End_indices,
            batch_size, num_spins, num_high_order_term,
            d_coeff_high_order_term, d_coeff_one_order_term, d_energies);

        find_min_energy_index << <1, 1 >> > (d_energies, batch_size, d_min_index);

        // Get results
        int h_min_index;
        cudaMemcpy(&h_min_index, d_min_index, sizeof(int), cudaMemcpyDeviceToHost);

        vector<int> h_x_sign_optimal(num_spins);
        cudaMemcpy(h_x_sign_optimal.data(), d_x_sign + h_min_index * num_spins, num_spins * sizeof(int), cudaMemcpyDeviceToHost);

        // Output results
        cout << "solution: ";
        for (int i = 0; i < h_x_sign_optimal.size(); ++i)
        {
            cout << (1 - h_x_sign_optimal[i]) / 2;
            if (i < h_x_sign_optimal.size() - 1)
            {
                cout << ", ";
            }
        }
        cout << endl;

        time_i++;

        // Free memory
        cudaFree(d_x);
        cudaFree(d_e);
        cudaFree(d_x_sign);
        cudaFree(d_high_order_term_result);
        cudaFree(d_gradient_term);
        cudaFree(d_energies);
        cudaFree(d_min_index);
        cudaFree(d_Spin_label);
        cudaFree(d_Spin_group_gradient);
        cudaFree(d_coeff_high_order_term);
        cudaFree(d_coeff_one_order_term);
        cudaFree(d_Start_indices);
        cudaFree(d_End_indices);
        cudaFree(d_Start_indices_gradient);
        cudaFree(d_End_indices_gradient);
    }
    return 0;
}
