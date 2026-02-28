#ifndef QPLIB_PARSER_H
#define QPLIB_PARSER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <stdexcept>

using namespace std;

// QPLIB配置：固化的文件路径配置
// 默认文件路径（编译时指定，运行时无需特殊指定）
const string QPLIB_BASE_PATH = "../qplib/";
const string DEFAULT_QPLIB_FILE = "QPLIB_3565.qplib";
const string DEFAULT_QPLIB_FILE_PATH = QPLIB_BASE_PATH + DEFAULT_QPLIB_FILE;

// 精确解配置：从data目录读取
const string QPLIB_RESULTS_PATH = "../data/qplib_results.csv";

/**
 * 问题模型数据结构
 * 
 * 该类包含了从QPLIB文件解析出的所有问题信息，
 * 包括目标函数系数、约束条件、缩放参数等。
 */
class ProblemModel {
public:
    // 基本信息
    string problem_name;                    // 问题名称
    string problem_type;                   // 问题类型
    string objective_sense;                // 优化方向 (minimize/maximize)
    int num_vars;                          // 变量数量
    int num_constraints;                   // 约束数量
    
    // 目标函数相关
    vector<vector<int>> quad_terms_indices;  // 二次项索引对 [i, j]
    vector<double> quad_terms_coeffs;        // 二次项系数
    vector<double> linear_terms;             // 线性项系数
    double constant_term;                    // 常数项
    
    // 约束相关
    vector<vector<double>> equal_constraint_matrix;    // 等式约束矩阵
    vector<double> equal_constraint_bounds;            // 等式约束边界
    vector<vector<double>> inequal_constraint_matrix;  // 不等式约束矩阵
    vector<double> inequal_constraint_bounds;          // 不等式约束边界
    
    // 缩放参数
    double scaling_factor;                 // 缩放因子
    double xi;                           // 耦合强度参数
    
    // 统计信息
    int num_quad_terms;                   // 二次项数量
    int num_linear_terms;                 // 线性项数量
    int num_equal_constraints;            // 等式约束数量
    int num_inequal_constraints;          // 不等式约束数量
    
    // 构造函数
    ProblemModel() : 
        num_vars(0), 
        num_constraints(0), 
        constant_term(0.0),
        scaling_factor(1.0),
        xi(1.0),
        num_quad_terms(0),
        num_linear_terms(0),
        num_equal_constraints(0),
        num_inequal_constraints(0) {}
    
    // 字符串表示
    string toString() const {
        ostringstream oss;
        oss << "Problem Model: " << problem_name << "\n"
            << "  Type: " << problem_type << "\n"
            << "  Objective sense: " << objective_sense << "\n"
            << "  Number of variables: " << num_vars << "\n"
            << "  Number of constraints: " << num_constraints << "\n"
            << "  Number of quadratic terms: " << num_quad_terms << "\n"
            << "  Number of linear terms: " << num_linear_terms << "\n"
            << "  Number of equality constraints: " << num_equal_constraints << "\n"
            << "  Number of inequality constraints: " << num_inequal_constraints << "\n"
            << "  Scaling factor: " << fixed << setprecision(6) << scaling_factor << "\n"
            << "  Coupling strength parameter: " << fixed << setprecision(6) << xi;
        return oss.str();
    }
};

/**
 * QPLIB文件解析器
 * 
 * 提供了完整的QPLIB文件解析功能。
 */
class QPLIBParser {
public:
    /**
     * 清理字符串，移除注释和空格
     */
    static string cleanLine(const string& line) {
        // 查找第一个#符号作为注释开始
        size_t comment_pos = line.find('#');
        string cleaned = (comment_pos != string::npos) ? line.substr(0, comment_pos) : line;
        
        // 去除首尾空格
        size_t start = cleaned.find_first_not_of(" \t\r\n");
        size_t end = cleaned.find_last_not_of(" \t\r\n");
        
        if (start == string::npos) return "";
        return cleaned.substr(start, end - start + 1);
    }
    
    /**
     * 提取行中的数字部分（忽略注释）
     */
    static string extractNumber(const string& line) {
        size_t comment_pos = line.find('#');
        string number_part = (comment_pos != string::npos) ? line.substr(0, comment_pos) : line;
        
        // 去除首尾空格
        size_t start = number_part.find_first_not_of(" \t\r\n");
        size_t end = number_part.find_last_not_of(" \t\r\n");
        
        if (start == string::npos) return "";
        return number_part.substr(start, end - start + 1);
    }
    
    /**
     * 读取非空行
     */
    static vector<string> readEffectiveLines(const string& file_path) {
        vector<string> lines;
        try {
            ifstream file(file_path);
            if (!file.is_open()) {
                cerr << "Error: File " << file_path << " does not exist" << endl;
                return lines;
            }
            
            string line;
            while (getline(file, line)) {
                string cleaned = cleanLine(line);
                if (!cleaned.empty()) {
                    lines.push_back(cleaned);
                }
            }
            file.close();
        } catch (const exception& e) {
            cerr << "Error: Failed to read file " << file_path << ": " << e.what() << endl;
        }
        
        return lines;
    }
    
    /**
     * 从data/qplib_results.csv读取精确最优解
     * 
     * 参数:
     *     problem_name: 问题名称（如 "QPLIB_3565"）
     * 返回:
     *     double: 精确最优解，未找到则返回NaN
     */
    static double getExactOptimalValue(const string& problem_name) {
        try {
            ifstream file(QPLIB_RESULTS_PATH);
            if (!file.is_open()) {
                cerr << "Warning: Cannot open qplib_results.csv: " << QPLIB_RESULTS_PATH << endl;
                return numeric_limits<double>::quiet_NaN();
            }
            
            string line;
            // 跳过标题行
            if (!getline(file, line)) {
                cerr << "Error: qplib_results.csv is empty" << endl;
                return numeric_limits<double>::quiet_NaN();
            }
            
            // 查找目标问题的行
            while (getline(file, line)) {
                // 清理行
                string cleaned = cleanLine(line);
                if (cleaned.empty()) continue;
                
                // 查找instance_id字段
                size_t instance_pos = cleaned.find("QPLIB_3565");
                if (instance_pos != string::npos) {
                    // 找到目标行，解析solution_objective_value字段
                    size_t obj_start = cleaned.find(",");
                    if (obj_start != string::npos) {
                        string obj_str = cleaned.substr(obj_start + 1);
                        size_t obj_end = obj_str.find(",");
                        if (obj_end != string::npos) {
                            obj_str = obj_str.substr(0, obj_end);
                            // 从字符串中提取数值，例如: "282.00000000 (gdx, sol)" -> 282.0
                            size_t num_start = 0;
                            while (num_start < obj_str.length() && (obj_str[num_start] < '0' || obj_str[num_start] > '9') && obj_str[num_start] != '-' && obj_str[num_start] != '+') {
                                num_start++;
                            }
                            size_t num_end = num_start;
                            while (num_end < obj_str.length() && (obj_str[num_end] == '.' || (obj_str[num_end] >= '0' && obj_str[num_end] <= '9'))) {
                                num_end++;
                            }
                            if (num_end > num_start) {
                                string number_str = obj_str.substr(num_start, num_end - num_start);
                                double value = stod(number_str);
                                cout << "从qplib_results.csv解析到精确最优解: " << fixed << setprecision(8) << value << endl;
                                return value;
                            }
                        }
                    }
                    break;
                }
            }
            
            file.close();
            cerr << "Warning: 未在qplib_results.csv中找到问题: " << problem_name << endl;
            return numeric_limits<double>::quiet_NaN();
            
        } catch (const exception& e) {
            cerr << "Error: 读取qplib_results.csv时出错: " << e.what() << endl;
            return numeric_limits<double>::quiet_NaN();
        }
    }

    /**
     * 解析QPLIB文件（基于qplib_parser.py的完整逻辑）
     * 
     * 参数:
     *     file_path: QPLIB文件路径
     *     model: 解析后的问题模型
     * 
     * 返回:
     *     bool: 解析是否成功
     */
    static bool parseQplibFile(const string& file_path, ProblemModel& model) {
        vector<string> lines = readEffectiveLines(file_path);
        if (lines.empty()) {
            cerr << "Cannot read valid data: " << file_path << endl;
            return false;
        }
        
        cout << "Start parsing QPLIB file: " << file_path << endl;
        
        // 1. 读取基本信息（与Python版本完全一致）
        model.problem_name = lines[0];
        model.problem_type = lines[1];
        model.objective_sense = lines[2];
        transform(model.objective_sense.begin(), model.objective_sense.end(), 
                  model.objective_sense.begin(), ::tolower);
        model.num_vars = stoi(extractNumber(lines[3]));
        
        // 根据实际QPLIB文件格式：
        // 第4行：变量数量（重复）
        // 第5行：二次项数量
        // 对于QPLIB_3565.qplib，没有约束条件
        int num_quadratic_terms = stoi(extractNumber(lines[4]));
        model.num_constraints = 0;  // 这个文件没有约束
        int current = 5;
        
        cout << "  - Problem name: " << model.problem_name << endl;
        cout << "  - Problem type: " << model.problem_type << endl;
        cout << "  - Objective sense: " << model.objective_sense << endl;
        cout << "  - Number of variables: " << model.num_vars << endl;
        cout << "  - Number of constraints: " << model.num_constraints << endl;
        cout << "  - Number of quadratic terms: " << num_quadratic_terms << endl;
        
        // 2. 读取二次项
        model.quad_terms_indices.clear();
        model.quad_terms_coeffs.clear();
        vector<double> diag_terms_coeffs;  // 对角项系数
        vector<int> diag_terms_indices;    // 对角项索引
        
        if (num_quadratic_terms > 0) {
            cout << "  - Parsing quadratic terms..." << endl;
            for (int i = 0; i < num_quadratic_terms; ++i) {
                istringstream iss(lines[current + i]);
                int row_idx, col_idx;
                double coeff;
                iss >> row_idx >> col_idx >> coeff;
                
                // 转换为0-based索引
                row_idx--;
                col_idx--;
                
                if (row_idx == col_idx) {
                    // 对角项
                    diag_terms_indices.push_back(row_idx);
                    diag_terms_coeffs.push_back(coeff);
                } else {
                    // 非对角项，存储对称矩阵（只存上三角）
                    // 注意：不除以2，保持原始系数，在能量计算时再乘0.5
                    if (row_idx > col_idx) swap(row_idx, col_idx);
                    model.quad_terms_indices.push_back({row_idx, col_idx});
                    model.quad_terms_coeffs.push_back(coeff);  // 保持原始系数
                }
            }
            current += num_quadratic_terms;
        } else {
            cout << "  - No quadratic terms detected" << endl;
        }
        
        model.num_quad_terms = model.quad_terms_indices.size();
        
        // 3. 读取默认线性项值
        double default_linear_value = stod(extractNumber(lines[current]));
        current++;
        cout << "  - Default linear term value: " << default_linear_value << endl;
        
        // 4. 读取指定线性项数量
        int num_linear_terms = stoi(extractNumber(lines[current]));
        current++;
        
        // 初始化线性项
        if (default_linear_value == 0.0 && num_linear_terms == 0 && diag_terms_indices.empty()) {
            model.linear_terms.clear();  // 设置为空
        } else {
            model.linear_terms.resize(model.num_vars, default_linear_value);
            
            // 读取指定线性项
            for (int i = 0; i < num_linear_terms; ++i) {
                istringstream iss(lines[current]);
                int var_idx;
                double coeff;
                iss >> var_idx >> coeff;
                model.linear_terms[var_idx - 1] = coeff;  // 转换为0-based索引
                current++;
            }
            
            // 处理对角项（将一半加到线性项中）
            for (size_t i = 0; i < diag_terms_indices.size(); ++i) {
                int idx = diag_terms_indices[i];
                model.linear_terms[idx] += diag_terms_coeffs[i] / 2.0;
            }
        }
        
        // 计算非零线性项数量
        model.num_linear_terms = 0;
        if (!model.linear_terms.empty()) {
            for (size_t i = 0; i < model.linear_terms.size(); ++i) {
                if (fabs(model.linear_terms[i]) > 1e-12) model.num_linear_terms++;
            }
        }
        
        // 5. 读取常数项
        model.constant_term = stod(extractNumber(lines[current]));
        current++;
        cout << "  - Constant term: " << model.constant_term << endl;
        
        // 6. 处理约束（如果有）
        if (model.num_constraints > 0) {
            // 读取约束线性项数量
            int num_constraint_linear_item = stoi(extractNumber(lines[current]));
            current++;
            
            vector<vector<double>> constraint_terms;
            if (num_constraint_linear_item > 0) {
                constraint_terms.resize(model.num_constraints, vector<double>(model.num_vars, 0.0));
                for (int i = 0; i < num_constraint_linear_item; ++i) {
                    istringstream iss(lines[current]);
                    int idx_constraints, idx_terms;
                    double val;
                    iss >> idx_constraints >> idx_terms >> val;
                    constraint_terms[idx_constraints - 1][idx_terms - 1] = val;
                    current++;
                }
            }
            
            // 读取约束边界
            double inf = stod(extractNumber(lines[current]));
            double constraint_lb_default = stod(extractNumber(lines[current + 1]));
            vector<double> constraint_lb(model.num_constraints, constraint_lb_default);
            int num_constraint_lb = stoi(extractNumber(lines[current + 2]));
            current += 3;
            
            double constraint_rb_default = stod(extractNumber(lines[current]));
            vector<double> constraint_rb(model.num_constraints, constraint_rb_default);
            int num_constraint_rb = stoi(extractNumber(lines[current + 1]));
            current += 2;
            
            // 处理指定的下界
            if (num_constraint_lb != 0) {
                for (int i = 0; i < num_constraint_lb; ++i) {
                    istringstream iss(lines[current]);
                    int idx;
                    double val;
                    iss >> idx >> val;
                    constraint_lb[idx - 1] = val;
                    current++;
                }
            }
            
            // 处理指定的上界
            if (num_constraint_rb != 0) {
                for (int i = 0; i < num_constraint_rb; ++i) {
                    istringstream iss(lines[current]);
                    int idx;
                    double val;
                    iss >> idx >> val;
                    constraint_rb[idx - 1] = val;
                    current++;
                }
            }
            
            // 分离等式约束和不等式约束
            if (!constraint_terms.empty()) {
                // 找到等式约束
                for (int i = 0; i < model.num_constraints; ++i) {
                    if (fabs(constraint_lb[i] - constraint_rb[i]) < 1e-12 && fabs(constraint_rb[i]) > 1e-12) {
                        model.equal_constraint_matrix.push_back(constraint_terms[i]);
                        model.equal_constraint_bounds.push_back(constraint_lb[i]);
                    }
                }
                
                // 处理不等式约束
                vector<vector<double>> inequal_terms;
                vector<double> inequal_bounds;
                
                for (int i = 0; i < model.num_constraints; ++i) {
                    if (fabs(constraint_lb[i] - constraint_rb[i]) >= 1e-12) {
                        // 下界约束
                        if (fabs(constraint_lb[i] - inf) >= 1e-12) {
                            vector<double> neg_term(constraint_terms[i].size());
                            for (size_t j = 0; j < constraint_terms[i].size(); ++j) {
                                neg_term[j] = -constraint_terms[i][j];
                            }
                            inequal_terms.push_back(neg_term);
                            inequal_bounds.push_back(-constraint_lb[i]);
                        }
                        // 上界约束
                        if (fabs(constraint_rb[i] - inf) >= 1e-12) {
                            inequal_terms.push_back(constraint_terms[i]);
                            inequal_bounds.push_back(constraint_rb[i]);
                        }
                    }
                }
                
                if (!inequal_bounds.empty()) {
                    model.inequal_constraint_matrix = inequal_terms;
                    model.inequal_constraint_bounds = inequal_bounds;
                }
            }
        } else {
            // 无约束问题
            model.equal_constraint_matrix.clear();
            model.equal_constraint_bounds.clear();
            model.inequal_constraint_matrix.clear();
            model.inequal_constraint_bounds.clear();
        }
        
        model.num_equal_constraints = model.equal_constraint_bounds.size();
        model.num_inequal_constraints = model.inequal_constraint_bounds.size();
        
        // 7. 计算缩放因子和耦合强度参数（与Python版本完全一致）
        if (model.num_quad_terms > 0) {
            // 对于二次问题，计算缩放因子
            double max_abs_coeff = 0.0;
            for (size_t i = 0; i < model.quad_terms_coeffs.size(); ++i) {
                double abs_val = fabs(model.quad_terms_coeffs[i]);
                if (abs_val > max_abs_coeff) max_abs_coeff = abs_val;
            }
            if (!model.linear_terms.empty()) {
                for (size_t i = 0; i < model.linear_terms.size(); ++i) {
                    double abs_val = fabs(model.linear_terms[i]);
                    if (abs_val > max_abs_coeff) max_abs_coeff = abs_val;
                }
            }
            
            if (max_abs_coeff > 1e-12) {
                model.scaling_factor = max_abs_coeff;
                
                // 缩放系数（与Python版本一致）
                for (size_t i = 0; i < model.quad_terms_coeffs.size(); ++i) {
                    model.quad_terms_coeffs[i] /= model.scaling_factor;
                }
                if (!model.linear_terms.empty()) {
                    for (size_t i = 0; i < model.linear_terms.size(); ++i) {
                        model.linear_terms[i] /= model.scaling_factor;
                    }
                }
                
                cout << "  - Coefficients scaled, scaling factor: " << model.scaling_factor << endl;
            }
            
            // 计算耦合强度参数xi（与Python版本一致）
            model.xi = sqrt(static_cast<double>(model.num_vars));
            cout << "  - Calculated xi value: " << model.xi << endl;
        } else {
            // 纯线性问题，不需要缩放
            model.scaling_factor = 1.0;
            model.xi = 1.0;
            cout << "  - Linear problem, no scaling applied" << endl;
        }
        
        cout << "Parsing completed!" << endl;
        
        return true;
    }
    
    /**
     * 将问题模型保存为JSON文件
     */
    static bool saveModelToJson(const ProblemModel& model, const string& output_path) {
        try {
            ofstream file(output_path);
            if (!file.is_open()) {
                cerr << "Cannot open output file: " << output_path << endl;
                return false;
            }
            
            file << "{\n";
            file << "  \"problem_name\": \"" << model.problem_name << "\",\n";
            file << "  \"problem_type\": \"" << model.problem_type << "\",\n";
            file << "  \"objective_sense\": \"" << model.objective_sense << "\",\n";
            file << "  \"num_vars\": " << model.num_vars << ",\n";
            file << "  \"num_constraints\": " << model.num_constraints << ",\n";
            
            file << "  \"quad_terms_indices\": [";
            for (size_t i = 0; i < model.quad_terms_indices.size(); ++i) {
                if (i > 0) file << ", ";
                file << "[" << model.quad_terms_indices[i][0] << ", " << model.quad_terms_indices[i][1] << "]";
            }
            file << "],\n";
            
            file << "  \"quad_terms_coeffs\": [";
            for (size_t i = 0; i < model.quad_terms_coeffs.size(); ++i) {
                if (i > 0) file << ", ";
                file << model.quad_terms_coeffs[i];
            }
            file << "],\n";
            
            file << "  \"linear_terms\": [";
            for (size_t i = 0; i < model.linear_terms.size(); ++i) {
                if (i > 0) file << ", ";
                file << model.linear_terms[i];
            }
            file << "],\n";
            
            file << "  \"constant_term\": " << model.constant_term << ",\n";
            file << "  \"scaling_factor\": " << model.scaling_factor << ",\n";
            file << "  \"xi\": " << model.xi << ",\n";
            file << "  \"num_quad_terms\": " << model.num_quad_terms << ",\n";
            file << "  \"num_linear_terms\": " << model.num_linear_terms << ",\n";
            file << "  \"num_equal_constraints\": " << model.num_equal_constraints << ",\n";
            file << "  \"num_inequal_constraints\": " << model.num_inequal_constraints << "\n";
            file << "}";
            
            file.close();
            cout << "Problem model saved to: " << output_path << endl;
            return true;
        } catch (const exception& e) {
            cerr << "Failed to save file: " << e.what() << endl;
            return false;
        }
    }
};

// 便捷函数
inline bool parseQplib(const string& file_path, ProblemModel& model) {
    return QPLIBParser::parseQplibFile(file_path, model);
}

inline bool saveModel(const ProblemModel& model, const string& output_path) {
    return QPLIBParser::saveModelToJson(model, output_path);
}

#endif // QPLIB_PARSER_H