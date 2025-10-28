#include <iostream>
#include <fstream>
#include <vector>
#include "json.hpp"
#include <chrono> 
#include <random>
#include <algorithm>
#include <limits>
#include <immintrin.h>
#include <cmath>

using namespace std;
using json = nlohmann::json;
using namespace std::chrono;

inline int sign(double x) 
{
    return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

struct BatchData 
{
    vector<double> x;
    vector<double> e;  // CAC-specific: amplitude control variable
    vector<int> x_sign;
    vector<double> poly_term;
    vector<double> gradient_term;
    double energy;

    BatchData(int num_spins, int num_high_order) :
        x(num_spins), e(num_spins), x_sign(num_spins),
        poly_term(num_high_order), gradient_term(num_spins), energy(0.0) 
    {
        // Initialize e to 1
        fill(e.begin(), e.end(), 1.0);
    }
};

class CACHUBOSolver 
{
private:
    vector<vector<int>> h_high_order_term;
    vector<int> h_one_order_term;
    vector<double> h_coeff_high_order_term;
    vector<double> h_coeff_one_order_term;
    int num_spins;
    int num_high_order_term;

    vector<vector<int>> var_to_terms;
    vector<vector<int>> term_positions;

public:
    CACHUBOSolver(const vector<vector<int>>& J, const vector<double>& coeff) 
    {
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
        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            mt19937 local_gen(seed + batch);
            uniform_real_distribution<double> dis(0.0, 1e-4);  // CAC uses smaller initial values

            for (int i = 0; i < num_spins; ++i) 
            {
                batches[batch].x[i] = dis(local_gen);
                batches[batch].e[i] = 1.0;  // Initialize to 1
            }
        }
    }

    void calculate_poly_term_batch(vector<BatchData>& batches) 
    {
        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            for (int i = 0; i < num_high_order_term; i++) 
            {
                double temp = 1.0;
                for (int var_idx : term_positions[i]) 
                {
                    temp *= batches[batch].x[var_idx];
                }
                batches[batch].poly_term[i] = temp * h_coeff_high_order_term[i];
            }
        }
    }

    void calculate_gradient_batch(vector<BatchData>& batches) 
    {
        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            for (int i = 0; i < num_spins; i++) 
            {
                batches[batch].gradient_term[i] = h_coeff_one_order_term[i];
            }

            for (int i = 0; i < num_spins; i++) 
            {
                if (batches[batch].x[i] != 0) 
                {
                    for (int term_idx : var_to_terms[i]) 
                    {
                        batches[batch].gradient_term[i] +=
                            batches[batch].poly_term[term_idx] / batches[batch].x[i];
                    }
                }
            }
        }
    }

    void update_cac_batch(vector<BatchData>& batches, double p, double xi, double dt,
        double alpha, double beta) 
    {
        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            for (int i = 0; i < num_spins; i++) 
            {
                double& x = batches[batch].x[i];
                double& e = batches[batch].e[i];
                double grad = batches[batch].gradient_term[i];

                // CAC update equation: x = x + (-x^3 + (p-1)*x + xi*e*grad_val)*dt
                x += (-x * x * x + (p - 1.0) * x + xi * e * grad) * dt;

                // e update equation: e = e + (-beta * e * (x^2 - alpha)) * dt
                e += (-beta * e * (x * x - alpha)) * dt;
            }
        }
    }

    void apply_cac_constraints(vector<BatchData>& batches, double alpha) 
    {
        double limit = 1.5 * sqrt(alpha);

        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            for (int i = 0; i < num_spins; i++) 
            {
                double& x = batches[batch].x[i];

                if (abs(x) > limit) 
                {
                    x = (x > 0) ? limit : -limit;
                }
            }
        }
    }

    double calc_energy_single(const vector<int>& x_sign) 
    {
        double energy = 0.0;

        for (int i = 0; i < num_high_order_term; i++) 
        {
            int temp = 1;
            for (int var_idx : term_positions[i]) 
            {
                temp *= x_sign[var_idx];
            }
            energy += temp * h_coeff_high_order_term[i];
        }

        for (int i = 0; i < num_spins; i++) 
        {
            energy += x_sign[i] * h_coeff_one_order_term[i];
        }

        return -energy;
    }

    int find_best_solution(vector<BatchData>& batches) 
    {
        for (int batch = 0; batch < batches.size(); ++batch) 
        {
            for (int i = 0; i < num_spins; ++i) 
            {
                batches[batch].x_sign[i] = sign(batches[batch].x[i]);
            }
            batches[batch].energy = calc_energy_single(batches[batch].x_sign);
        }

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
        // CAC parameter setup
        double sum_coeff_square = 0.0;
        for (double c : h_coeff_high_order_term) {
            sum_coeff_square += c * c;
        }
        double xi = sqrt(2.0 * num_spins / sum_coeff_square);
        double dt = 0.075;
        int Tr = static_cast<int>(0.9 * max_iters);

        double beta = 0.3;

        // Generate p and alpha sequences
        vector<double> p_values(max_iters);
        vector<double> alpha_values(max_iters);

        for (int i = 0; i < Tr; i++) 
        {
            p_values[i] = -0.5 + 1.5 * i / (Tr - 1);
            alpha_values[i] = 1.0 + 2.0 * i / (Tr - 1);
        }
        for (int i = Tr; i < max_iters; i++) 
        {
            p_values[i] = 1.0;
            alpha_values[i] = 3.0;
        }

        vector<BatchData> batches(batch_size, BatchData(num_spins, num_high_order_term));
        init_batch_data(batches, time(nullptr));

        auto start_time = high_resolution_clock::now();

        for (int iters = 0; iters < max_iters; iters++) 
        {
            calculate_poly_term_batch(batches);
            calculate_gradient_batch(batches);
            update_cac_batch(batches, p_values[iters], xi, dt, alpha_values[iters], beta);
            apply_cac_constraints(batches, alpha_values[iters]);
        }

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        cout << "CAC total execution time: " << duration.count() / 1000.0 << " ms" << endl;

        int best_batch = find_best_solution(batches);

        vector<int> result(num_spins);
        for (int i = 0; i < num_spins; ++i) 
        {
            result[i] = (batches[best_batch].x_sign[i] - 1) / -2;
        }

        cout << "CAC best solution from batch " << best_batch << endl;
        return result;
    }

    int get_num_spins() const { return num_spins; }
};

int count_diff(const vector<int>& a, const vector<int>& b, int size) 
{
    int count = 0;
    for (int i = 0; i < size; ++i) 
    {
        if (a[i] != b[i]) {
            ++count;
        }
    }
    return count;
}

int main() 
{
    cout << "Using CAC algorithm serial version" << endl;

    int time_i = 0;

    while (true) 
    {
        string filePath = "D:/HW/CUDAcode/DEMO/demo_data/" + to_string(time_i) + ".json";

        ifstream file(filePath);
        if (!file.is_open()) 
        {
            break;
        }
        cout << "CAC DATA" << time_i << endl;

        json data;
        file >> data;
        file.close();

        vector<vector<int>> J = data["J"];
        vector<double> coeff = data["coeff"];
        vector<int> h_label = data["label"];

        CACHUBOSolver solver(J, coeff);
        

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
