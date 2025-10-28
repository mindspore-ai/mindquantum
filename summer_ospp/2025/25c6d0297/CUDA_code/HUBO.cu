#include "HUBO.h" // Include the header file

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <time.h>

#include <cuda_runtime.h>  // CUDA runtime library
#include <curand_kernel.h>

using namespace std;
using namespace std::chrono;

// CUDA error checking macro is defined in HUBO.h

// ==================== CUDA Kernel Functions ====================

// Get device properties and calculate optimal configuration
DeviceConfig getDeviceConfig() {
    DeviceConfig config;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    config.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    config.maxSharedMemPerBlock = prop.sharedMemPerBlock;
    config.warpSize = prop.warpSize;

    // Set conservative optimal block size to ensure sufficient shared memory
    config.optimalBlockSize = min(512, config.maxThreadsPerBlock);
    while (config.optimalBlockSize * sizeof(double) > config.maxSharedMemPerBlock / 2) {
        config.optimalBlockSize /= 2;
    }
    config.optimalBlockSize = max(config.optimalBlockSize, config.warpSize);

    config.maxBlocks = prop.multiProcessorCount * 8; // Conservative estimate

    return config;
}

// Random initialization kernel function
__global__ void initial_DSB(double* array, int batch_size, int num_spins, uint64_t seed)
{
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx < batch_size && spin_idx < num_spins)
    {
        int global_idx = batch_idx * num_spins + spin_idx;
        curandState state;
        curand_init(seed, global_idx, 0, &state);
        float random = curand_uniform(&state);
        array[global_idx] = random * 0.02 - 0.01; // 0.01 * normal distribution approximation
    }
}

// Sign function kernel
__global__ void sign(double* array, int* sign_array, int batch_size, int num_spins)
{
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx < batch_size && spin_idx < num_spins) {
        int global_idx = batch_idx * num_spins + spin_idx;
        sign_array[global_idx] = (array[global_idx] > 0) ? 1 : ((array[global_idx] < 0) ? -1 : 0);
    }
}

template<typename T>
__global__ void calculate_poly_term(T* d_x_data, int* Spin_label, int* start_indices, int* end_indices,
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
                poly_term *= d_x_data[spin_global_idx];
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

template<typename T>
__global__ void calculate_gradient_term(T* d_x_data, int* Spin_group, int* start_indices, int* end_indices,
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
        int data_idx = batch_idx * num_spins + spin_idx;

        if (start != -1 && end != -1)
        {
            for (int i = start; i <= end; ++i)
            {
                int energy_idx = batch_idx * num_groups + Spin_group[i];
                // Note: Need to handle division by zero
                if (d_x_data[data_idx] != 0)
                {
                    grad += partial_energy[energy_idx] / d_x_data[data_idx];
                }
            }
        }
        gradient[gradient_idx] = grad;
    }
}

__global__ void update_BSB(double P, double xi, double* d_x, int batch_size, int num_spins,
    double dt, double delta, double* gradient_term, double* array_y)
{
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx < batch_size && spin_idx < num_spins)
    {
        int global_idx = batch_idx * num_spins + spin_idx;
        array_y[global_idx] += (-(delta - P) * d_x[global_idx] + xi * gradient_term[global_idx]) * dt;
        d_x[global_idx] += array_y[global_idx] * dt * delta;
        if (abs(d_x[global_idx]) > 1.0)
        {
            d_x[global_idx] = (d_x[global_idx] > 0) ? 1.0 : -1.0;
            array_y[global_idx] = 0.0;
        }
    }
}

// Update function (DSB)
__global__ void update_DSB(double P, double xi, double* d_x, int batch_size, int num_spins,
    double dt, double delta, double* gradient_term, double* array_y)
{
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx < batch_size && spin_idx < num_spins)
    {
        int global_idx = batch_idx * num_spins + spin_idx;
        array_y[global_idx] += (-(delta - P) * d_x[global_idx] + xi * gradient_term[global_idx]) * dt;
        d_x[global_idx] += array_y[global_idx] * dt * delta;
        if (abs(d_x[global_idx]) > 1.0)
        {
            d_x[global_idx] = (d_x[global_idx] > 0) ? 1.0 : -1.0;
            array_y[global_idx] = 0.0;
        }
    }
}


// ==================== QAIA Class Implementation ====================

// Add scope resolution 'QAIA::'
QAIA::QAIA(const vector<vector<int>>& J, const vector<double>& coeff, int iter, int b_size, const string& be)
{
    backend = be;
    n_iter = iter;
    batch_size = b_size;
    
    // Initialize all pointers to nullptr
    d_x = nullptr;
    d_x_sign = nullptr;
    d_Spin_label = nullptr;
    d_Spin_group_gradient = nullptr;
    d_coeff_high_order_term = nullptr;
    d_coeff_one_order_term = nullptr;
    d_Start_indices = nullptr;
    d_End_indices = nullptr;
    d_Start_indices_gradient = nullptr;
    d_End_indices_gradient = nullptr;
    d_high_order_term_result = nullptr;
    d_gradient_term = nullptr;

    if (backend == "CPU")
    {
        for (const auto& row : J)
        {
            if (row.size() > 1)
            {
                J_high_order.push_back(row);
            }
            else
            {
                J_one_order.push_back(row[0]);
            }
        }

        num_high_order_term = J_high_order.size();
        num_spins = J_one_order.size();

        x.resize(batch_size, vector<double>(num_spins, 0.0));
        x_sign.resize(batch_size, vector<int>(num_spins, 0));
        coeff_high_order.resize(num_high_order_term);
        coeff_one_order.resize(num_spins);
        energy_batch.resize(batch_size, 0.0);

        coeff_high_order.assign(coeff.begin(), coeff.begin() + num_high_order_term);
        coeff_one_order.assign(coeff.begin() + num_high_order_term, coeff.end());
    }
    else if (backend == "GPU")
    {
        for (const auto& row : J)
        {
            if (row.size() > 1)
            {
                J_high_order.push_back(row);
            }
            else
            {
                J_one_order.push_back(row[0]);
            }
        }

        num_high_order_term = J_high_order.size();
        num_spins = J_one_order.size();

        x.resize(batch_size, vector<double>(num_spins, 0.0));
        x_sign.resize(batch_size, vector<int>(num_spins, 0));
        coeff_high_order.resize(num_high_order_term);
        coeff_one_order.resize(num_spins);
        energy_batch.resize(batch_size, 0.0);

        coeff_high_order.assign(coeff.begin(), coeff.begin() + num_high_order_term);
        coeff_one_order.assign(coeff.begin() + num_high_order_term, coeff.end());

        for (int time_a = 0; time_a < J_high_order.size(); time_a++)
        {
            for (int time_b = 0; time_b < J_high_order[time_a].size(); time_b++)
            {
                h_Spin_label.push_back(J_high_order[time_a][time_b]);
                h_Spin_group.push_back(time_a);
            }
        }

        num_Spin_label = h_Spin_label.size();
        h_Start_indices.resize(num_high_order_term, -1);
        h_End_indices.resize(num_high_order_term, -1);
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

        h_Start_indices_gradient.resize(num_spins, -1);
        h_End_indices_gradient.resize(num_spins, -1);

        vector<size_t> indices(h_Spin_label.size());
        for (size_t time_d = 0; time_d < indices.size(); ++time_d)
        {
            indices[time_d] = time_d;
        }

        sort(indices.begin(), indices.end(), [this](size_t time_e, size_t time_f)
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
                if (current_group_1 != -1) {
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

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_x, batch_size * num_spins * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_x_sign, batch_size * num_spins * sizeof(int)));

        CHECK_CUDA_ERROR(cudaMalloc(&d_Spin_label, num_Spin_label * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_Spin_group_gradient, num_Spin_label * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_coeff_high_order_term, num_high_order_term * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_coeff_one_order_term, num_spins * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_Start_indices, num_high_order_term * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_End_indices, num_high_order_term * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_Start_indices_gradient, num_spins * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_End_indices_gradient, num_spins * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_high_order_term_result, batch_size * num_high_order_term * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_gradient_term, batch_size * num_spins * sizeof(double)));

        // Copy data to device (non-batch data)
        CHECK_CUDA_ERROR(cudaMemcpy(d_Spin_label, h_Spin_label.data(), num_Spin_label * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_Spin_group_gradient, h_Spin_group_gradient.data(), num_Spin_label * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_coeff_high_order_term, coeff_high_order.data(), num_high_order_term * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_coeff_one_order_term, coeff_one_order.data(), num_spins * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_Start_indices, h_Start_indices.data(), num_high_order_term * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_End_indices, h_End_indices.data(), num_high_order_term * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_Start_indices_gradient, h_Start_indices_gradient.data(), num_spins * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_End_indices_gradient, h_End_indices_gradient.data(), num_spins * sizeof(int), cudaMemcpyHostToDevice));
    }
    else
    {
        cout << "error!" << endl;
    }
}

// Add scope resolution 'QAIA::'
void QAIA::caculate_energy()
{
    for (int b = 0; b < batch_size; ++b)
    {
        // Ensure poly_term and gradient_term are sized correctly if backend is CPU
        if (poly_term.size() != num_high_order_term) poly_term.resize(num_high_order_term, 0.0);
        if (gradient_term.size() != num_spins) gradient_term.resize(num_spins, 0.0);
        
        fill(poly_term.begin(), poly_term.end(), 0.0);
        fill(gradient_term.begin(), gradient_term.end(), 0.0);

        // Calculate sign
        for (int i = 0; i < num_spins; ++i)
        {
            x_sign[b][i] = (x[b][i] > 0) ? 1 : ((x[b][i] < 0) ? -1 : 0);
        }

        // Calculate total energy
        energy_batch[b] = 0.0; // Reset energy for this batch
        for (int j = 0; j < num_high_order_term; ++j)
        {
            int temp = 1;
            for (int k = 0; k < J_high_order[j].size(); k++)
            {
                temp *= x_sign[b][J_high_order[j][k]];
            }
            poly_term[j] = temp * coeff_high_order[j];
            energy_batch[b] += poly_term[j];
        }
        for (int k = 0; k < num_spins; ++k)
        {
            energy_batch[b] += coeff_one_order[k] * x_sign[b][k];
        }
    }
}

// Add scope resolution 'QAIA::'
void QAIA::sign_single(vector<double>& array_x, vector<int>& array_x_sign)
{
    for (int i = 0; i < array_x.size(); ++i)
    {
        array_x_sign[i] = (array_x[i] > 0) ? 1 : ((array_x[i] < 0) ? -1 : 0);
    }
}

// Add scope resolution 'QAIA::'
void QAIA::sign_batch(vector<vector<double>>& array_x, vector<vector<int>>& array_x_sign)
{
    for (int b = 0; b < array_x.size(); ++b)
    {
        for (int i = 0; i < array_x[b].size(); ++i)
        {
            array_x_sign[b][i] = (array_x[b][i] > 0) ? 1 : ((array_x[b][i] < 0) ? -1 : 0);
        }
    }
}


// ==================== BSB Class Implementation ====================

// Add scope resolution 'BSB::' and call base constructor
BSB::BSB(const vector<vector<int>>& J, const vector<double>& coeff, int iter, int b_size, const string& be)
    : QAIA(J, coeff, iter, b_size, be)  // Call parent class constructor
{
    d_y = nullptr; // Initialize pointer
    poly_term.resize(num_high_order_term, 0.0);
    gradient_term.resize(num_spins, 0.0);
    y.resize(batch_size, vector<double>(num_spins, 0.0));

    if (backend == "GPU")
    {
        CHECK_CUDA_ERROR(cudaMalloc(&d_y, batch_size * num_spins * sizeof(double)));
        // Initialize thread configuration
        blockSize = dim3(256);
        gridSize_batch_spins = dim3(batch_size, (num_spins + blockSize.x - 1) / blockSize.x);
        gridSize_batch_groups = dim3(batch_size, (num_high_order_term + blockSize.x - 1) / blockSize.x);
    }
}

// Add scope resolution 'BSB::'
void BSB::initial()
{
    if (backend == "CPU")
    {
        srand(time(NULL));
        for (int b = 0; b < batch_size; ++b)
        {
            for (int n = 0; n < num_spins; ++n)
            {
                x[b][n] = (rand() / (double)RAND_MAX) * 0.02 - 0.01;
                y[b][n] = (rand() / (double)RAND_MAX) * 0.02 - 0.01;
            }
        }
    }
    else if (backend == "GPU")
    {
        DeviceConfig config = getDeviceConfig();
        cout << "Device Config - Max Threads: " << config.maxThreadsPerBlock
            << ", Optimal Block Size: " << config.optimalBlockSize
            << ", Max Shared Mem: " << config.maxSharedMemPerBlock << " bytes" << endl;

        // Use the grid/block sizes initialized in the constructor
        initial_DSB << <gridSize_batch_spins, blockSize >> > (d_x, batch_size, num_spins, time(NULL));
        initial_DSB << <gridSize_batch_spins, blockSize >> > (d_y, batch_size, num_spins, time(NULL) + 1);

        for (int b = 0; b < batch_size; ++b)
        {
            CHECK_CUDA_ERROR(cudaMemcpy(x[b].data(), d_x + b * num_spins, num_spins * sizeof(double), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(y[b].data(), d_y + b * num_spins, num_spins * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }
}

// Add scope resolution 'BSB::'
void BSB::update()
{
    if (backend == "CPU")
    {
        auto start_time = high_resolution_clock::now();

        for (int iters = 0; iters < n_iter; iters++)
        {
            P = (double)iters / (n_iter - 1);  // Linear increase from 0 to 1
            // Update for each batch
            for (int b = 0; b < batch_size; ++b)
            {
                fill(poly_term.begin(), poly_term.end(), 0.0);
                fill(gradient_term.begin(), gradient_term.end(), 0.0);

                // Calculate gradient
                for (int j = 0; j < num_high_order_term; ++j)
                {
                    double temp = 1.0; // Use double for intermediate product
                    for (int k = 0; k < J_high_order[j].size(); k++)
                    {
                        temp *= x[b][J_high_order[j][k]];
                    }
                    poly_term[j] = temp * coeff_high_order[j];
                }

                for (int i = 0; i < num_spins; i++)
                {
                    for (int j = 0; j < num_high_order_term; j++)
                    {
                        for (int k = 0; k < J_high_order[j].size(); k++)
                        {
                            if (J_high_order[j][k] == i)
                            {
                                // Handle potential division by zero
                                if (x[b][i] != 0.0)
                                {
                                    gradient_term[i] += poly_term[j] / x[b][i];
                                }
                            }
                        }
                    }
                    gradient_term[i] += coeff_one_order[i];
                }

                // Update variables
                for (int i = 0; i < num_spins; i++)
                {
                    y[b][i] += ((P - 1) * x[b][i] + xi * gradient_term[i]) * dt;
                }
                for (int i = 0; i < num_spins; i++)
                {
                    x[b][i] += y[b][i] * dt;
                }

                for (int i = 0; i < num_spins; i++)
                {
                    if (x[b][i] > 1.0)
                    {
                        x[b][i] = 1.0;
                        y[b][i] = 0;
                    }
                    else if (x[b][i] < -1.0)
                    {
                        x[b][i] = -1.0;
                        y[b][i] = 0;
                    }
                }
            }
        }

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        cout << "Solving time: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    else if (backend == "GPU")
    {
        auto start_time = high_resolution_clock::now();

        for (int iters = 0; iters < n_iter; ++iters)
        {
            P = double(iters) / double(n_iter - 1);

            calculate_poly_term << <gridSize_batch_groups, blockSize >> > (d_x, d_Spin_label, d_Start_indices, d_End_indices,
                batch_size, num_spins, num_high_order_term,
                d_coeff_high_order_term, d_high_order_term_result);

            calculate_gradient_term << <gridSize_batch_spins, blockSize >> > (d_x, d_Spin_group_gradient,
                d_Start_indices_gradient, d_End_indices_gradient,
                batch_size, num_spins, num_high_order_term,
                d_high_order_term_result, d_coeff_one_order_term, d_gradient_term);

            update_BSB << <gridSize_batch_spins, blockSize >> > (P, xi, d_x, batch_size, num_spins, dt, delta, d_gradient_term, d_y);
        }

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        cout << "Solving time: " << duration.count() / 1000.0 << " milliseconds" << endl;

        for (int b = 0; b < batch_size; ++b)
        {
            CHECK_CUDA_ERROR(cudaMemcpy(x[b].data(), d_x + b * num_spins, num_spins * sizeof(double), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(y[b].data(), d_y + b * num_spins, num_spins * sizeof(double), cudaMemcpyDeviceToHost));
        }

        // Free device memory
        cudaFree(d_x);
        cudaFree(d_x_sign);
        cudaFree(d_Spin_label);
        cudaFree(d_Spin_group_gradient);
        cudaFree(d_coeff_high_order_term);
        cudaFree(d_coeff_one_order_term);
        cudaFree(d_Start_indices);
        cudaFree(d_End_indices);
        cudaFree(d_Start_indices_gradient);
        cudaFree(d_End_indices_gradient);
        cudaFree(d_high_order_term_result);
        cudaFree(d_gradient_term);
        cudaFree(d_y);
    }

    // Output the batch with lowest energy
    caculate_energy();
    double min_energy = energy_batch[0];
    opt_x = 0; // Reset opt_x
    for (int b = 0; b < batch_size; ++b)
    {
        if (energy_batch[b] < min_energy)
        {
            min_energy = energy_batch[b];
            opt_x = b;
        }
    }
    cout << "opt x:" << endl;
    cout << "[" << endl;
    for (double val : x[opt_x]) // Use double for loop
    {
        // Map from {-1, 1} to {0, 1}
        cout << (val - 1.0) / -2.0 << " "; 
    }
    cout << "]" << endl;
}

// Add scope resolution 'BSB::'
std::vector<double> BSB::get_solution() const
{
    return x[opt_x];
}

// Add scope resolution 'BSB::'
double BSB::get_energy() const
{
    return energy_batch[opt_x];
}

// ==================== DSB Class Implementation ====================

// Add scope resolution 'DSB::' and call base constructor
DSB::DSB(const vector<vector<int>>& J, const vector<double>& coeff, int iter, int b_size, const string& be)
    : QAIA(J, coeff, iter, b_size, be)  // Call parent class constructor
{
    d_y = nullptr; // Initialize pointer
    poly_term.resize(num_high_order_term, 0.0);
    gradient_term.resize(num_spins, 0.0);
    y.resize(batch_size, vector<double>(num_spins, 0.0));

    if (backend == "GPU")
    {
        CHECK_CUDA_ERROR(cudaMalloc(&d_y, batch_size * num_spins * sizeof(double)));
        // Initialize thread configuration
        blockSize = dim3(256);
        gridSize_batch_spins = dim3(batch_size, (num_spins + blockSize.x - 1) / blockSize.x);
        gridSize_batch_groups = dim3(batch_size, (num_high_order_term + blockSize.x - 1) / blockSize.x);
    }
}

// Add scope resolution 'DSB::'
void DSB::initial()
{
    if (backend == "CPU")
    {
        srand(time(NULL));
        for (int b = 0; b < batch_size; ++b)
        {
            for (int n = 0; n < num_spins; ++n)
            {
                x[b][n] = (rand() / (double)RAND_MAX) * 0.02 - 0.01;
                y[b][n] = (rand() / (double)RAND_MAX) * 0.02 - 0.01;
            }
        }
    }
    else if (backend == "GPU")
    {
        DeviceConfig config = getDeviceConfig();
        cout << "Device Config - Max Threads: " << config.maxThreadsPerBlock
            << ", Optimal Block Size: " << config.optimalBlockSize
            << ", Max Shared Mem: " << config.maxSharedMemPerBlock << " bytes" << endl;
        
        // Use the grid/block sizes initialized in the constructor
        initial_DSB << <gridSize_batch_spins, blockSize >> > (d_x, batch_size, num_spins, time(NULL));
        initial_DSB << <gridSize_batch_spins, blockSize >> > (d_y, batch_size, num_spins, time(NULL) + 1);

        for (int b = 0; b < batch_size; ++b)
        {
            CHECK_CUDA_ERROR(cudaMemcpy(x[b].data(), d_x + b * num_spins, num_spins * sizeof(double), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(y[b].data(), d_y + b * num_spins, num_spins * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }
}

// Add scope resolution 'DSB::'
void DSB::update()
{
    if (backend == "CPU")
    {
        auto start_time = high_resolution_clock::now();

        for (int iters = 0; iters < n_iter; iters++)
        {
            P = (double)iters / (n_iter - 1);  // Linear increase from 0 to 1
            // Update for each batch
            for (int b = 0; b < batch_size; ++b)
            {
                fill(poly_term.begin(), poly_term.end(), 0.0);
                fill(gradient_term.begin(), gradient_term.end(), 0.0);

                // Calculate sign
                for (int i = 0; i < num_spins; ++i)
                {
                    x_sign[b][i] = (x[b][i] > 0) ? 1 : ((x[b][i] < 0) ? -1 : 0);
                }

                // Calculate gradient
                for (int j = 0; j < num_high_order_term; ++j)
                {
                    int temp = 1;
                    for (int k = 0; k < J_high_order[j].size(); k++)
                    {
                        temp *= x_sign[b][J_high_order[j][k]];
                    }
                    poly_term[j] = temp * coeff_high_order[j];
                }

                for (int i = 0; i < num_spins; i++)
                {
                    for (int j = 0; j < num_high_order_term; j++)
                    {
                        for (int k = 0; k < J_high_order[j].size(); k++)
                        {
                            if (J_high_order[j][k] == i)
                            {
                                // Handle division by zero (though sign should be -1 or 1)
                                if (x_sign[b][i] != 0)
                                {
                                    gradient_term[i] += poly_term[j] / x_sign[b][i];
                                }
                            }
                        }
                    }
                    gradient_term[i] += coeff_one_order[i];
                }

                // Update variables
                for (int i = 0; i < num_spins; i++)
                {
                    y[b][i] += ((P - 1) * x[b][i] + xi * gradient_term[i]) * dt;
                }
                for (int i = 0; i < num_spins; i++)
                {
                    x[b][i] += y[b][i] * dt;
                }

                for (int i = 0; i < num_spins; i++)
                {
                    if (x[b][i] > 1.0)
                    {
                        x[b][i] = 1.0;
                        y[b][i] = 0;
                    }
                    else if (x[b][i] < -1.0)
                    {
                        x[b][i] = -1.0;
                        y[b][i] = 0;
                    }
                }
            }
        }

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        cout << "Solving time: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    else if (backend == "GPU")
    {
        auto start_time = high_resolution_clock::now();

        for (int iters = 0; iters < n_iter; ++iters)
        {
            P = double(iters) / double(n_iter - 1);

            sign << <gridSize_batch_spins, blockSize >> > (d_x, d_x_sign, batch_size, num_spins);

            calculate_poly_term << <gridSize_batch_groups, blockSize >> > (d_x_sign, d_Spin_label, d_Start_indices, d_End_indices,
                batch_size, num_spins, num_high_order_term,
                d_coeff_high_order_term, d_high_order_term_result);

            calculate_gradient_term << <gridSize_batch_spins, blockSize >> > (d_x_sign, d_Spin_group_gradient,
                d_Start_indices_gradient, d_End_indices_gradient,
                batch_size, num_spins, num_high_order_term,
                d_high_order_term_result, d_coeff_one_order_term, d_gradient_term);

            update_DSB << <gridSize_batch_spins, blockSize >> > (P, xi, d_x, batch_size, num_spins, dt, delta, d_gradient_term, d_y);
        }

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        cout << "Solving time: " << duration.count() / 1000.0 << " milliseconds" << endl;

        for (int b = 0; b < batch_size; ++b)
        {
            CHECK_CUDA_ERROR(cudaMemcpy(x[b].data(), d_x + b * num_spins, num_spins * sizeof(double), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(y[b].data(), d_y + b * num_spins, num_spins * sizeof(double), cudaMemcpyDeviceToHost));
        }

        // Free device memory
        cudaFree(d_x);
        cudaFree(d_x_sign);
        cudaFree(d_Spin_label);
        cudaFree(d_Spin_group_gradient);
        cudaFree(d_coeff_high_order_term);
        cudaFree(d_coeff_one_order_term);
        cudaFree(d_Start_indices);
        cudaFree(d_End_indices);
        cudaFree(d_Start_indices_gradient);
        cudaFree(d_End_indices_gradient);
        cudaFree(d_high_order_term_result);
        cudaFree(d_gradient_term);
        cudaFree(d_y);
    }

    // Output the batch with lowest energy
    caculate_energy();
    double min_energy = energy_batch[0];
    opt_x = 0; // Reset opt_x
    for (int b = 0; b < batch_size; ++b)
    {
        if (energy_batch[b] < min_energy)
        {
            min_energy = energy_batch[b];
            opt_x = b;
        }
    }
    cout << "opt x:" << endl;
    cout << "[" << endl;
    for (double val : x[opt_x]) // Use double for loop
    {
        // Map from {-1, 1} to {0, 1}
        cout << (val - 1.0) / -2.0 << " ";
    }
    cout << "]" << endl;
}

// Add scope resolution 'DSB::'
std::vector<double> DSB::get_solution() const
{
    return x[opt_x];
}

// Add scope resolution 'DSB::'
double DSB::get_energy() const
{
    return energy_batch[opt_x];
}

// Remove the main() function
/*
int main()
{

    //BSB bsb(J, coeff, 100, 10, "CPU");
    //bsb.initial();
    //bsb.update();

    return 0;
}
*/