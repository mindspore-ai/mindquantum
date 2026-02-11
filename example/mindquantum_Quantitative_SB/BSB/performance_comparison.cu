#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <regex>

using namespace std;
using namespace std::chrono;

// 结构体存储性能结果
struct PerformanceResult {
    string name;
    double energy;
    double time_ms;
    bool success;
    string data_type;  // 数据类型标识
};

// CSV数据结构
struct CSVData {
    string instance_id;
    double solution_objective_value;
    string sol_file_path;
    string detail_url;
    string sol_url;
};

// 解析CSV文件，返回instance_id到solution_objective_value的映射
map<string, double> parse_csv_data(const string& csv_file) {
    map<string, double> data;
    ifstream file(csv_file);
    string line;
    
    if (!file.is_open()) {
        cout << "Warning: Could not open CSV file: " << csv_file << endl;
        return data;
    }
    
    // 跳过标题行
    getline(file, line);
    
    while (getline(file, line)) {
        // 解析CSV行
        stringstream ss(line);
        string field;
        vector<string> fields;
        
        // 简单CSV解析（处理引号内的逗号）
        bool in_quotes = false;
        string current_field;
        for (char c : line) {
            if (c == '"') {
                in_quotes = !in_quotes;
            } else if (c == ',' && !in_quotes) {
                fields.push_back(current_field);
                current_field.clear();
            } else {
                current_field += c;
            }
        }
        fields.push_back(current_field);
        
        if (fields.size() >= 2) {
            string instance_id = fields[0];
            string obj_value_str = fields[1];
            
            // 使用正则表达式提取数值
            regex number_regex(R"((-?\d+\.?\d*))");
            smatch match;
            
            if (regex_search(obj_value_str, match, number_regex)) {
                double obj_value = stod(match[1].str());
                data[instance_id] = obj_value;
            }
        }
    }
    
    file.close();
    return data;
}

// 解析输出结果，提取能量值和运行时间
PerformanceResult parse_output(const string& filename, const string& solver_name, const string& data_type) {
    ifstream file(filename);
    string line;
    PerformanceResult result;
    result.name = solver_name;
    result.data_type = data_type;
    result.energy = 0.0;
    result.time_ms = 0.0;
    result.success = false;

    // 优先查找最新的输出格式 "BSB GPU OBJECTIVE VALUE"
    // 其次查找 "BSB OBJECTIVE VALUE"
    // 最后查找 "BEST: Optimal objective value:"
    bool found_energy = false;

    while (getline(file, line)) {
        // 提取最优能量值（按优先级顺序）
        if (!found_energy) {
            // 优先级1: "BSB GPU OBJECTIVE VALUE:" (最新GPU版本)
            size_t pos = line.find("BSB GPU OBJECTIVE VALUE:");
            if (pos != string::npos) {
                string energy_str = line.substr(pos + 25);
                energy_str.erase(0, energy_str.find_first_not_of(" \t"));
                energy_str.erase(energy_str.find_last_not_of(" \t\r\n") + 1);
                try {
                    result.energy = stod(energy_str);
                    result.success = true;
                    found_energy = true;
                    cout << "    Parsed energy (GPU FINAL): " << result.energy << " from string: '" << energy_str << "'" << endl;
                } catch (const exception& e) {
                    cout << "    WARNING: Failed to parse GPU FINAL energy value: " << e.what() << endl;
                }
                continue;
            }

            // 优先级2: "BSB OBJECTIVE VALUE:" (CPU版本)
            pos = line.find("BSB OBJECTIVE VALUE:");
            if (pos != string::npos) {
                string energy_str = line.substr(pos + 20);
                energy_str.erase(0, energy_str.find_first_not_of(" \t"));
                energy_str.erase(energy_str.find_last_not_of(" \t\r\n") + 1);
                try {
                    result.energy = stod(energy_str);
                    result.success = true;
                    found_energy = true;
                    cout << "    Parsed energy (CPU FINAL): " << result.energy << " from string: '" << energy_str << "'" << endl;
                } catch (const exception& e) {
                    cout << "    WARNING: Failed to parse CPU FINAL energy value: " << e.what() << endl;
                }
                continue;
            }

            // 优先级3: "BEST: Optimal objective value:" (备选方案)
            pos = line.find("BEST: Optimal objective value:");
            if (pos != string::npos) {
                string energy_str = line.substr(pos + 28);
                energy_str.erase(0, energy_str.find_first_not_of(" \t"));
                energy_str.erase(energy_str.find_last_not_of(" \t\r\n") + 1);
                try {
                    result.energy = stod(energy_str);
                    result.success = true;
                    found_energy = true;
                    cout << "    Parsed energy (BEST): " << result.energy << " from string: '" << energy_str << "'" << endl;
                } catch (const exception& e) {
                    cout << "    WARNING: Failed to parse BEST energy value: " << e.what() << endl;
                }
                continue;
            }
        }

        // 提取运行时间
        if (result.time_ms == 0.0 && line.find("TIME: Solving completed, total time:") != string::npos) {
            size_t pos = line.find("time:") + 5;
            string time_str = line.substr(pos);
            time_str.erase(0, time_str.find_first_not_of(" \t"));
            size_t end_pos = time_str.find(" ms");
            if (end_pos != string::npos) {
                time_str = time_str.substr(0, end_pos);
            }
            try {
                result.time_ms = stod(time_str);
                cout << "    Parsed time: " << result.time_ms << " ms" << endl;
            } catch (const exception& e) {
                cout << "    WARNING: Failed to parse time: " << e.what() << endl;
            }
        }
    }

    return result;
}

// 编译并运行指定的BSB求解器
bool run_solver(const string& solver_name, const string& cu_file, const string& exe_name, 
                const string& data_type, PerformanceResult& result) {
    cout << "Building and running " << solver_name << " (" << data_type << ")..." << endl;
    
    // 构建命令
    string build_cmd = "nvcc -diag-suppress 192 -o " + exe_name + " " + cu_file + " -lcublas -lcurand";
    
    // 使用qplib_parser.h中指定的默认路径
    string run_cmd = ".\\" + exe_name + " > " + exe_name + "_output.txt 2>&1";
    
    // 编译
    cout << "  Compiling " << cu_file << "..." << endl;
    int build_result = system(build_cmd.c_str());
    if (build_result != 0) {
        cout << "  ERROR: Failed to compile " << cu_file << endl;
        return false;
    }
    
    // 运行
    cout << "  Running " << exe_name << "..." << endl;
    auto start = high_resolution_clock::now();
    int run_result = system(run_cmd.c_str());
    auto end = high_resolution_clock::now();
    
    if (run_result == 0) {
        // 解析输出
        result = parse_output(exe_name + "_output.txt", solver_name, data_type);
        if (result.success) {
            cout << "  SUCCESS: Energy = " << result.energy << ", Time = " << result.time_ms << " ms" << endl;
            return true;
        } else {
            cout << "  WARNING: Could not extract energy from output" << endl;
            
            // 显示输出文件内容以便调试
            cout << "  Output file content: " << endl;
            ifstream output_file(exe_name + "_output.txt");
            string line;
            if (output_file.is_open()) {
                while (getline(output_file, line)) {
                    cout << "    " << line << endl;
                }
                output_file.close();
            }
            
            return false;
        }
    } else {
        cout << "  ERROR: Failed to run " << exe_name << endl;
        return false;
    }
}

// 打印性能比较表格
void print_comparison_table(const vector<PerformanceResult>& results, const map<string, double>& csv_data) {
    cout << "\n========================================" << endl;
    cout << "      BSB Solver Performance Table     " << endl;
    cout << "========================================" << endl;
    cout << left << setw(20) << "Solver" << setw(10) << "Type" << setw(15) << "Energy" << setw(15) << "Time (ms)" << setw(20) << "Q" << endl;
    cout << string(80, '-') << endl;
    
    // 找到最佳能量值
    double best_energy = results[0].energy;
    for (const auto& res : results) {
        if (res.success && res.energy < best_energy) {
            best_energy = res.energy;
        }
    }
    
    // 找到最快时间
    double best_time = results[0].time_ms;
    for (const auto& res : results) {
        if (res.success && res.time_ms > 0 && res.time_ms < best_time) {
            best_time = res.time_ms;
        }
    }
    
    // 打印每个结果
    for (const auto& res : results) {
        cout << left << setw(20) << res.name;
        cout << setw(10) << res.data_type;
        
        if (res.success) {
            cout << setw(15) << res.energy;
            cout << setw(15) << res.time_ms;
            
            // 新的Efficiency计算：energy / solution_objective_value
            double efficiency = 0.0;
            
            // 假设实例名称在结果名称中或使用默认值QPLIB_3565
            string instance_name = "QPLIB_3565"; // 默认值，可以根据实际情况调整
            
            // 尝试从结果名称中提取实例名
            size_t qplib_pos = res.name.find("QPLIB_");
            if (qplib_pos != string::npos) {
                size_t end_pos = res.name.find("_", qplib_pos + 6);
                if (end_pos != string::npos) {
                    instance_name = res.name.substr(0, end_pos);
                } else {
                    instance_name = res.name;
                }
            }
            
            auto it = csv_data.find(instance_name);
            if (it != csv_data.end()) {
                double solution_objective_value = it->second;
                
                // 避免除零错误
                if (abs(solution_objective_value) > 1e-10) {
                    efficiency = res.energy / solution_objective_value;
                    
                    // 如果是最大化问题且目标值为负，调整efficiency计算
                    if (solution_objective_value < 0 && res.energy < 0) {
                        efficiency = res.energy / solution_objective_value; // 两个负数相除得正数
                    }
                } else {
                    efficiency = (res.energy == 0) ? 1.0 : 0.0;
                }
            } else {
                // 如果找不到CSV数据，使用传统方法作为备选
                double energy_ratio = (best_energy < 0) ? (res.energy / best_energy) : 1.0;
                double time_ratio = (best_time > 0) ? (best_time / res.time_ms) : 1.0;
                efficiency = energy_ratio * time_ratio;
                cout << "(CSV missing)";
            }
            
            cout << setw(20) << fixed << setprecision(4) << efficiency;
        } else {
            cout << setw(15) << "FAILED";
            cout << setw(15) << "N/A";
            cout << setw(20) << "N/A";
        }
        cout << endl;
    }
    cout << endl;
    
    // 详细分析
    cout << "Detailed Analysis:" << endl;
    cout << "- Best Energy: " << best_energy << " (more negative is better)" << endl;
    cout << "- Fastest Time: " << best_time << " ms" << endl;
    cout << "- New Efficiency: Energy / Solution_Objective_Value (from CSV)" << endl;
    cout << "  Values closer to 1.0 indicate performance closer to optimal solution" << endl;
    cout << "  CSV Data Source: qplib_results.csv (solution_objective_value)" << endl;
}

// 按数据类型分组分析
void print_type_analysis(const vector<PerformanceResult>& results, const map<string, double>& csv_data) {
    cout << "\n========================================" << endl;
    cout << "        Data Type Analysis             " << endl;
    cout << "========================================" << endl;
    
    map<string, vector<PerformanceResult>> type_groups;
    for (const auto& res : results) {
        type_groups[res.data_type].push_back(res);
    }
    
    for (const auto& group : type_groups) {
        cout << "\n" << group.first << " Data Type:" << endl;
        cout << string(30, '-') << endl;
        
        double avg_energy = 0.0;
        double avg_time = 0.0;
        int success_count = 0;
        
        for (const auto& res : group.second) {
            if (res.success) {
                avg_energy += res.energy;
                avg_time += res.time_ms;
                success_count++;
            }
        }
        
        if (success_count > 0) {
            avg_energy /= success_count;
            avg_time /= success_count;
            cout << "  Average Energy: " << avg_energy << endl;
            cout << "  Average Time: " << avg_time << " ms" << endl;
            cout << "  Success Rate: " << success_count << "/" << group.second.size() << endl;
        } else {
            cout << "  All implementations failed" << endl;
        }
    }
}

// 内存效率分析
void print_memory_analysis(const vector<PerformanceResult>& results) {
    cout << "\n========================================" << endl;
    cout << "        Memory Efficiency Analysis      " << endl;
    cout << "========================================" << endl;
    
    cout << "Data Type Memory Efficiency (relative to float32):" << endl;
    cout << "- FP8: 4x memory reduction" << endl;
    cout << "- INT8: 4x memory reduction" << endl;
    cout << "- FP16: 2x memory reduction" << endl;
    cout << "- Mixed Precision: Varies by implementation" << endl;
    cout << endl;
    
    // 分析每个数据类型的最佳表现
    map<string, PerformanceResult> best_by_type;
    for (const auto& res : results) {
        if (res.success) {
            if (best_by_type.find(res.data_type) == best_by_type.end() || 
                res.energy < best_by_type[res.data_type].energy) {
                best_by_type[res.data_type] = res;
            }
        }
    }
    
    cout << "Best Performance by Data Type:" << endl;
    for (const auto& best : best_by_type) {
        cout << "- " << best.first << ": " << best.second.name 
             << " (Energy: " << best.second.energy 
             << ", Time: " << best.second.time_ms << " ms)" << endl;
    }
}

int main() {
    cout << "========================================" << endl;
    cout << "      BSB Solver Performance Comparison   " << endl;
    cout << "========================================" << endl << endl;
    
    // 定义要测试的BSB求解器
    vector<tuple<string, string, string>> solvers = {
        {"BSB_CPU_fp16", "BSB_CPU_fp16.cu", "FP16"},
        {"BSB_CPU_fp8", "BSB_CPU_fp8.cu", "FP8"},
        {"BSB_CPU_int8", "BSB_CPU_int8.cu", "INT8"},
        {"BSB_GPU_fp16", "BSB_GPU_fp16.cu", "FP16"},
        {"BSB_GPU_fp8", "BSB_GPU_fp8.cu", "FP8"},
        {"BSB_GPU_int8", "BSB_GPU_int8.cu", "INT8"}
    };
    
    vector<PerformanceResult> results;
    
    // 运行每个求解器
    for (const auto& solver : solvers) {
        PerformanceResult result;
        string solver_name = get<0>(solver);
        string cu_file = get<1>(solver);
        string data_type = get<2>(solver);
        string exe_name = solver_name;
        
        bool success = run_solver(solver_name, cu_file, exe_name, data_type, result);
        if (success) {
            results.push_back(result);
        } else {
            // 即使失败也添加结果，标记为失败
            result.name = solver_name;
            result.data_type = data_type;
            result.success = false;
            results.push_back(result);
        }
        
        cout << endl;
    }
    
    // 加载CSV数据
    cout << "Loading CSV data from qplib_results.csv..." << endl;
    map<string, double> csv_data = parse_csv_data("../data/qplib_results.csv");
    cout << "Loaded " << csv_data.size() << " instances from CSV" << endl;
    
    // 打印比较结果
    print_comparison_table(results, csv_data);
    
    // 数据类型分析
    print_type_analysis(results, csv_data);
    
    // 内存效率分析
    print_memory_analysis(results);
    
    // 推荐最佳求解器
    cout << "\n========================================" << endl;
    cout << "              Recommendations           " << endl;
    cout << "========================================" << endl;
    
    PerformanceResult best_energy_solver = results[0];
    PerformanceResult best_time_solver = results[0];
    PerformanceResult best_overall = results[0];
    
    for (const auto& res : results) {
        if (res.success) {
            if (res.energy < best_energy_solver.energy) {
                best_energy_solver = res;
            }
            if (res.time_ms > 0 && res.time_ms < best_time_solver.time_ms) {
                best_time_solver = res;
            }
        }
    }
    
    // 计算综合效率
    double best_efficiency = 0.0;
    for (const auto& res : results) {
        if (res.success) {
            double energy_ratio = (best_energy_solver.energy < 0) ? (res.energy / best_energy_solver.energy) : 1.0;
            double time_ratio = (best_time_solver.time_ms > 0) ? (best_time_solver.time_ms / res.time_ms) : 1.0;
            double efficiency = energy_ratio * time_ratio;
            
            if (efficiency > best_efficiency) {
                best_efficiency = efficiency;
                best_overall = res;
            }
        }
    }
    
    cout << "Best Energy Performance: " << best_energy_solver.name 
         << " (" << best_energy_solver.data_type << ")"
         << " (Energy: " << best_energy_solver.energy << ")" << endl;
    cout << "Fastest Execution: " << best_time_solver.name 
         << " (" << best_time_solver.data_type << ")"
         << " (Time: " << best_time_solver.time_ms << " ms)" << endl;
    cout << "Best Overall Performance: " << best_overall.name 
         << " (" << best_overall.data_type << ")" << endl;
    
    // 特定场景推荐
    cout << "\nScenario-based Recommendations:" << endl;
    cout << "- For Maximum Memory Efficiency: Choose FP8/INT8 implementations" << endl;
    cout << "- For Best Numerical Precision: Choose FP16 implementations" << endl;
    cout << "- For Ultra-Low Memory Footprint: Choose FP8 implementations" << endl;
    cout << "- For GPU Acceleration: Choose GPU implementations" << endl;
    cout << "- For Balanced Performance: Choose mixed precision implementations" << endl;
    
    cout << "\n========================================" << endl;
    
    return 0;
}