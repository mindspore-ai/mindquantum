/**
 * @file sb_cuda_gset_reader.h
 * @brief Gset 数据文件读取函数
 *
 * 参考 SB_solve_Gset.py 中的 read_gset() 函数实现
 * 支持 Gset 格式文件的读取和转换
 */

#ifndef SB_CUDA_GSET_READER_H
#define SB_CUDA_GSET_READER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Gset 数据结构（COO格式）
 *
 * 对应 Python 中的 COO 矩阵格式
 */
typedef struct {
    int num_vertices;          // 顶点数量
    int num_edges;             // 边数量
    int* row_indices;          // 行索引 [num_edges]
    int* col_indices;          // 列索引 [num_edges]
    float* weights;           // 边权重 [num_edges]
} GsetData;

/**
 * @brief 读取 Gset 文件
 *
 * 对应 Python 代码：
 * graph = pd.read_csv(filename, sep=' ')
 * n_v = int(graph.columns[0])
 * n_e = int(graph.columns[1])
 *
 * @param filename Gset 文件路径
 * @param negate 是否对权重取负（MAX-CUT 转 MIN-CUT）
 * @param data 输出的 Gset 数据
 * @return 0 成功，非0 失败
 */
int read_gset_file(const char* filename, int negate, GsetData* data) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return -1;
    }

    // 读取第一行：顶点数和边数
    int num_vertices, num_edges;
    if (fscanf(fp, "%d %d", &num_vertices, &num_edges) != 2) {
        std::cerr << "Error: Invalid Gset file format (header)" << std::endl;
        fclose(fp);
        return -2;
    }

    // 分配内存
    data->num_vertices = num_vertices;
    data->num_edges = num_edges;
    data->row_indices = (int*)malloc(num_edges * sizeof(int));
    data->col_indices = (int*)malloc(num_edges * sizeof(int));
    data->weights = (float*)malloc(num_edges * sizeof(float));

    if (data->row_indices == NULL || data->col_indices == NULL || data->weights == NULL) {
        std::cerr << "Error: Memory allocation failed" << std::endl;
        fclose(fp);
        return -3;
    }

    // 读取边数据
    int u, v;
    float w;
    int count = 0;
    while (fscanf(fp, "%d %d %f", &u, &v, &w) == 3) {
        if (count >= num_edges) {
            std::cerr << "Warning: More edges than expected (" << num_edges << ")" << std::endl;
            break;
        }

        // Gset 文件中的顶点索引从 1 开始，转换为从 0 开始
        data->row_indices[count] = u - 1;
        data->col_indices[count] = v - 1;

        // 处理权重（是否取负）
        data->weights[count] = negate ? -w : w;

        count++;
    }

    fclose(fp);

    // 验证边数量
    if (count != num_edges) {
        std::cerr << "Warning: Expected " << num_edges << " edges, read " << count << std::endl;
        // 更新实际的边数量
        data->num_edges = count;
    }

    return 0;
}

/**
 * @brief 将 Gset 数据转换为 BSB/DSB 求解器格式
 *
 * 对应 Python 代码：
 * G = coo_matrix((weights, (row_indices, col_indices)), shape=(n_v, n_v))
 *
 * 转换为：
 * - second_order_terms: 边索引 [num_edges * 2]
 * - second_order_coeff: 边系数 [num_edges]
 * - one_order_coeff: 线性系数 [num_vertices]
 *
 * @param gset 输入的 Gset 数据
 * @param second_order_terms 输出：二阶项索引 [num_edges * 2]
 * @param second_order_coeff 输出：二阶项系数 [num_edges]
 * @param one_order_coeff 输出：一阶项系数（初始化为0）
 * @return 0 成功，非0 失败
 */
int convert_gset_to_solver_format(const GsetData* gset,
                                  int* second_order_terms,
                                  float* second_order_coeff,
                                  float* one_order_coeff) {
    // 初始化一阶项系数为 0（Gset 问题没有外场）
    for (int i = 0; i < gset->num_vertices; ++i) {
        one_order_coeff[i] = 0.0f;
    }

    // 填充二阶项
    for (int i = 0; i < gset->num_edges; ++i) {
        second_order_terms[i * 2] = gset->row_indices[i];
        second_order_terms[i * 2 + 1] = gset->col_indices[i];
        second_order_coeff[i] = gset->weights[i];
    }

    return 0;
}

/**
 * @brief 释放 Gset 数据内存
 *
 * @param data Gset 数据
 */
void free_gset_data(GsetData* data) {
    if (data->row_indices != NULL) {
        free(data->row_indices);
        data->row_indices = NULL;
    }
    if (data->col_indices != NULL) {
        free(data->col_indices);
        data->col_indices = NULL;
    }
    if (data->weights != NULL) {
        free(data->weights);
        data->weights = NULL;
    }
}

/**
 * @brief 打印 Gset 数据信息
 *
 * @param data Gset 数据
 */
void print_gset_info(const GsetData* data) {
    std::cout << "Gset Data Information:" << std::endl;
    std::cout << "  Vertices: " << data->num_vertices << std::endl;
    std::cout << "  Edges: " << data->num_edges << std::endl;
    std::cout << "  Density: " << (2.0 * data->num_edges) / (data->num_vertices * (data->num_vertices - 1)) << std::endl;

    // 统计权重信息
    float min_weight = data->weights[0];
    float max_weight = data->weights[0];
    float sum_weight = 0.0f;

    for (int i = 0; i < data->num_edges; ++i) {
        float w = data->weights[i];
        if (w < min_weight) min_weight = w;
        if (w > max_weight) max_weight = w;
        sum_weight += w;
    }

    std::cout << "  Weight range: [" << min_weight << ", " << max_weight << "]" << std::endl;
    std::cout << "  Average weight: " << (sum_weight / data->num_edges) << std::endl;
}

#ifdef __cplusplus
}
#endif

#endif // SB_CUDA_GSET_READER_H
