#include <iostream>
#include <fstream>
#include <vector>
#include "json.hpp"
#include <chrono> 
#include <random>
#include <algorithm>
#include <limits>
#include <immintrin.h>

using namespace std;
using json = nlohmann::json;
using namespace std::chrono;

// Optimized inline function
inline int sign(double x) 
{
    return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

// Data structure for optimized data access
struct BatchData 
{
    vector<double> x;
    vector<double> y;
    vector<int> x_sign;
    vector<double> poly_term;
    vector<double> gradient_term;
    double energy;

    BatchData(int num_spins, int num_high_order) :
        x(num_spins), y(num_spins), x_sign(num_spins),
        poly_term(num_high_order), gradient_term(num_spins), energy(0.0) {}
};

class HUBOSolver 
{
private:
    vector<vector<int>> h_high_order_term;
    vector<int> h_one_order_term;
    vector<double> h_coeff_high_order_term;
    vector<double> h_coeff_one_order_term;
    int num_spins;
    int num_high_order_term;

    // Precomputed lookup tables
    vector<vector<int>> var_to_terms;  // Each variable's participating high-order term indices
    vector<vector<int>> term_positions;  // Variable positions within each high-order term

public:
    HUBOSolver(const vector<vector<int>>& J, const vector<double>& coeff) 
    {
        // Separate high-order and first-order terms
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

        num_high_order_term = h_high_order_term.size();
        num_spins = h_one_order_term.size();

        h_coeff_high_order_term.assign(coeff.begin(), coeff.begin() + num_high_order_term);
        h_coeff_one_order_term.assign(coeff.begin() + num_high_order_term, coeff.end());

        // Precompute lookup tables
        precompute_lookup_tables();
    }

    void precompute_lookup_tables() 
    {
        var_to_terms.resize(num_spins);
        term_positions.resize(num_high_order_term);

        for (int i = 0; i < num_high_order_term; i++) 
        {
            term_positions[i].resize(h_high_order_term[i].size());
            for (int j = 0; j < h_high_order_term[i].size(); j++) 
            {
                int var_idx = h_high_order_term[i][j];
                var_to_terms[var_idx].push_back(i);
                term_positions[i][j] = var_idx;
            }
        }
    }

    void init_batch_data(vector<BatchData>& batches, uint64_t seed) 
    {
        mt19937 gen(seed);
        uniform_real_distribution<double> dis(-0.01, 0.01);

        // Serial version without parallel processing
        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            mt19937 local_gen(seed + batch);
            uniform_real_distribution<double> local_dis(-0.01, 0.01);

            for (int i = 0; i < num_spins; ++i) 
            {
                batches[batch].x[i] = local_dis(local_gen);
                batches[batch].y[i] = local_dis(local_gen);
            }
        }
    }

    void sign_xy_batch(vector<BatchData>& batches) 
    {
        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            for (int i = 0; i < num_spins; ++i) 
            {
                batches[batch].x_sign[i] = sign(batches[batch].x[i]);
            }
        }
    }

    // Optimized polynomial term calculation - using precomputed lookup tables
    void calculate_poly_term_batch(vector<BatchData>& batches) 
    {
        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            for (int i = 0; i < num_high_order_term; i++) 
            {
                int temp = 1;
                // Use precomputed position information
                for (int var_idx : term_positions[i]) 
                {
                    temp *= batches[batch].x_sign[var_idx];
                }
                batches[batch].poly_term[i] = temp * h_coeff_high_order_term[i];
            }
        }
    }

    // Optimized gradient calculation - using precomputed lookup tables
    void calculate_gradient_batch(vector<BatchData>& batches) 
    {
        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            // Initialize with first-order term coefficients
            for (int i = 0; i < num_spins; i++) 
            {
                batches[batch].gradient_term[i] = h_coeff_one_order_term[i];
            }

            // Calculate high-order term contributions to gradient using lookup tables
            for (int i = 0; i < num_spins; i++) 
            {
                if (batches[batch].x_sign[i] != 0) 
                {
                    for (int term_idx : var_to_terms[i]) 
                    {
                        batches[batch].gradient_term[i] +=
                            batches[batch].poly_term[term_idx] / batches[batch].x_sign[i];
                    }
                }
            }
        }
    }

    void update_y_batch(vector<BatchData>& batches, double p, double xi, double dt, double delta) 
    {
        const double factor = -(delta - p);

        for (int batch = 0; batch < batches.size(); ++batch)
        {
            for (int i = 0; i < num_spins; i++) 
            {
                batches[batch].y[i] += (factor * batches[batch].x[i] +
                    xi * batches[batch].gradient_term[i]) * dt;
            }
        }
    }

    void update_x_batch(vector<BatchData>& batches, double dt, double delta) 
    {
        const double factor = dt * delta;

        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            // SIMD-optimized vector updates
            int i = 0;
#ifdef __AVX2__
            for (; i <= num_spins - 4; i += 4) 
            {
                __m256d x_vec = _mm256_loadu_pd(&batches[batch].x[i]);
                __m256d y_vec = _mm256_loadu_pd(&batches[batch].y[i]);
                __m256d factor_vec = _mm256_set1_pd(factor);
                __m256d result = _mm256_fmadd_pd(y_vec, factor_vec, x_vec);
                _mm256_storeu_pd(&batches[batch].x[i], result);
            }
#endif

            // Process remaining elements
            for (; i < num_spins; i++) 
            {
                batches[batch].x[i] += batches[batch].y[i] * factor;
            }
        }
    }

    void cond_xy_batch(vector<BatchData>& batches) 
    {
        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            for (int i = 0; i < num_spins; i++) 
            {
                if (abs(batches[batch].x[i]) > 1.0) 
                {
                    batches[batch].x[i] = (batches[batch].x[i] > 0) ? 1.0 : -1.0;
                    batches[batch].y[i] = 0.0;
                }
            }
        }
    }

    double calc_energy_single(const vector<int>& x_sign) 
    {
        double energy = 0.0;

        // Calculate high-order term energy
        for (int i = 0; i < num_high_order_term; i++) 
        {
            int temp = 1;
            for (int var_idx : term_positions[i]) 
            {
                temp *= x_sign[var_idx];
            }
            energy += temp * h_coeff_high_order_term[i];
        }

        // Calculate first-order term energy
        for (int i = 0; i < num_spins; i++) 
        {
            energy += x_sign[i] * h_coeff_one_order_term[i];
        }

        return -energy;
    }

    int find_best_solution(vector<BatchData>& batches) 
    {
        // Serial calculation of each batch's energy
        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            for (int i = 0; i < num_spins; ++i) 
            {
                batches[batch].x_sign[i] = sign(batches[batch].x[i]);
            }
            batches[batch].energy = calc_energy_single(batches[batch].x_sign);
        }

        // Find minimum energy
        int best_batch = 0;
        double min_energy = batches[0].energy;

        for (int batch = 1; batch < batches.size(); ++batch) 
        {
            if (batches[batch].energy < min_energy) 
            {
                min_energy = batches[batch].energy;
                best_batch = batch;
            }
        }

        return best_batch;
    }

    vector<int> solve(int max_iters = 100, int batch_size = 100) 
    {
        // Parameter settings
        double xi = 0.5;
        double dt = 1.0;
        double delta = 1.0;

        // Initialize batch data
        vector<BatchData> batches(batch_size, BatchData(num_spins, num_high_order_term));
        init_batch_data(batches, time(nullptr));

        auto start_time = high_resolution_clock::now();

        for (int iters = 0; iters < max_iters; iters++) 
        {
            double p = (double)iters / (max_iters - 1);

            sign_xy_batch(batches);
            calculate_poly_term_batch(batches);
            calculate_gradient_batch(batches);
            update_y_batch(batches, p, xi, dt, delta);
            update_x_batch(batches, dt, delta);
            cond_xy_batch(batches);
        }

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        cout << "Total execution time: " << duration.count() / 1000.0 << " ms" << endl;

        // Find best solution
        int best_batch = find_best_solution(batches);

        // Convert result
        vector<int> result(num_spins);
        for (int i = 0; i < num_spins; ++i) 
        {
            result[i] = (batches[best_batch].x_sign[i] - 1) / -2;
        }

        return result;
    }

    int get_num_spins() const { return num_spins; }
};


int main() 
{

    int time_i = 0;

    while (true) 
    {
        // Read data file
        string filePath = "demo_data/" + to_string(time_i) + ".json";
        
        ifstream file(filePath);
        if (!file.is_open()) 
        {
            cout << "Cannot open file: " << filePath << endl;
            return 1;
        }
        cout << "DATA" << time_i << endl;

        json data;
        file >> data;
        file.close();

        vector<vector<int>> J = data["J"];
        vector<double> coeff = data["coeff"];

        // Create solver
        HUBOSolver solver(J, coeff);

        // Solve
        vector<int> solution = solver.solve();
        cout << "solution: ";
        for (int i = 0; i < solution.size(); ++i)
        {
            cout << solution[i];
            if (i < solution.size() - 1)
            {
                cout << ", ";
            }
        }

        cout << endl;

        time_i++;
    }
    return 0;
}
