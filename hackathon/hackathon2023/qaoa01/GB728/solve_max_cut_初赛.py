from scipy.sparse import csr_matrix
import resource
from utils import *
import pickle
import math
import matplotlib.pyplot as plt
import random
import sys
import copy
import time
import numpy as np
from itertools import count
import os
os.environ['OMP_NUM_THREADS'] = '2'


'''
本赛题旨在引领选手探索，在NISQ时代规模有限的量子计算机上，求解真实场景中的大规模图分割问题。
本代码为求解最大割
'''
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (1024*1024*4000, hard))  # 4G
N_sub = 15  # 量子比特的规模限制

filelist = ['./graphs/regular_d3_n40_cut54_0.txt', './graphs/regular_d3_n80_cut108_0.txt',
            './graphs/weight_p0.5_n40_cut238.txt', './graphs/weight_p0.2_cut406.txt',
            './graphs/partition_n80_cut226_0.txt', './graphs/partition_n80_cut231_1.txt'
            ]
# filelist = ['./graphs/regular_d3_n80_cut108_0.txt']

# 参考了https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/2023/07_waikikilick/main.ipynb


class QaoaInQaoa(object):
    def __init__(self, _iteration):
        self.iteration = _iteration


    def split(self, g: nx.Graph, n_qubit, n_sub):
        sub_graphs = []

        c = nx.algorithms.community.greedy_modularity_communities(g, n_communities=n_sub)
        sub_list = [list(x) for x in c]
        for x in sub_list:
            if len(x) > n_qubit:
                n_ssub = math.ceil(len(x) / n_qubit)

                ssub_list = [x[n_qubit * i:n_qubit * (i + 1)] for i in range(n_ssub)]
                for i in range(n_ssub):
                    sub_graphs.append(ssub_list[i])
            else:
                sub_graphs.append(x)
        return sub_graphs

    def build_sub_qubo(self, solution, n_sub, J,heads, h=None, C=0.0):
        '''
        自定义函数选取subqubo子问题。
        例如可简单选择对cost影响最大的前n_sub个变量组合成为subqubo子问题。
        【注】本函数非必须，仅供参考

        返回
        subqubo问题变量的指标，和对应的J，h，C。
        '''
        delta_L = []
        for j in range(len(solution)):
            zj = solution[j]
            delta_score = 0.0
            for v in heads[j]:
                if solution[v] == zj:
                    delta_score += 1
                else:
                    delta_score -= 1
            delta_L.append(delta_score)

        sub_index = np.argpartition(delta_L, -n_sub)[-n_sub:] # subqubo子问题的变量们
        J_sub,h_sub,C_sub = calc_subqubo(sub_index, solution, J, h=h,C=C )
        return sub_index,J_sub,h_sub,C_sub

    def main(self, solution, g, origin_G):
        # 建立点->边对应关系
        heads = [[] for _ in range(len(solution))]
        for edge in g.edges:
            u, v = edge
            heads[u].append(v)
            heads[v].append(u)


       # 将图顶点随机分组，每组最大数不超过量子比特数。
        nodes = list(range(len(solution)))
        graphs_nodes = self.split(g, N_sub, math.ceil(len(g.nodes) / N_sub))
        node_graphs = [0 for _ in range(len(solution))]

        for i in range(len(graphs_nodes)):
            for node in graphs_nodes[i]:
                node_graphs[node] = i
        # # 根据子图顶点，挑出各个子图所包含的边
        total_graphs_edges = [[]
                              for _ in range(len(graphs_nodes))]  # 注意，可能有的子图可能没有内边
        for edge in g.edges:
            u = edge[0]
            v = edge[1]
            if node_graphs[u] == node_graphs[v]:
                total_graphs_edges[node_graphs[u]].append(edge)
         # 把没有边的子图拆掉
        bad_subgraphs = []
        bad_subgraphs_nodes = []
        for graphs_index in range(len(graphs_nodes)):
            if len(total_graphs_edges[graphs_index]) == 0:
                bad_subgraphs.append(graphs_index)
                for nd in graphs_nodes[graphs_index]:
                    bad_subgraphs_nodes.append(nd)
        for i in range(len(bad_subgraphs) - 1, -1, -1):
            pos = bad_subgraphs[i]
            graphs_nodes = graphs_nodes[:pos] + graphs_nodes[pos + 1:]
        for nd in bad_subgraphs_nodes:
            graphs_nodes.append([nd])
        for i in range(len(graphs_nodes)):
            for node in graphs_nodes[i]:
                node_graphs[node] = i


        # # 根据子图顶点，挑出各个子图所包含的边
        total_graphs_edges = [[]
                              for _ in range(len(graphs_nodes))]  # 注意，可能有的子图可能没有内边
        for edge in g.edges:
            u = edge[0]
            v = edge[1]
            if node_graphs[u] == node_graphs[v]:
                total_graphs_edges[node_graphs[u]].append(edge)

        # 将每个子图的顶点都重新映射为一个新图的顶点，这样就可以在量子计算中只使用很少的量子比特：
        rename_graphs_nodes = []
        rename_nodes_list_0 = []  # 新索引：顶点序号
        rename_nodes_list_1 = []  # 顶点序号：新索引

        for sub_graph_nodes in graphs_nodes:
            rename_graphs_nodes_0 = {}
            rename_graphs_nodes_1 = {}
            rename_node = []
            for (index, node) in enumerate(sub_graph_nodes):
                rename_node.append(index)
                rename_graphs_nodes_0[index] = node
                rename_graphs_nodes_1[node] = index
            rename_nodes_list_0.append(rename_graphs_nodes_0)
            rename_nodes_list_1.append(rename_graphs_nodes_1)
            rename_graphs_nodes.append(rename_node)

        # 根据重新映射的点，重新映射边
        rename_total_graphs_edges = []
        for (graphs_index, graphs_edges) in enumerate(total_graphs_edges):
            rename_graphs_edges = []
            for sub_graph_edges in graphs_edges:
                rename_graphs_edges.append(
                    (rename_nodes_list_1[graphs_index][sub_graph_edges[0]], rename_nodes_list_1[graphs_index][sub_graph_edges[1]]))
            rename_total_graphs_edges.append(rename_graphs_edges)

        # 采用 QAOA 计算各子图的最大割：
        graphs_num = len(graphs_nodes)  # 子图数量
        max_cuts = []
        left_nodes = []

        # 采用 QAOA 计算各子图的最大割：
        for graph_index in range(graphs_num):
            n = len(rename_graphs_nodes[graph_index])
            if len(rename_total_graphs_edges[graph_index]) == 0:
                sol = init_solution(n)
            else:
                edges = []
                for i in rename_total_graphs_edges[graph_index]:
                    edges.append((int(i[0]), int(i[1]), int(1)))
                edge_list = np.array(edges)
                # print(edge_list.shape)
                G = csr_matrix(
                    (-1 * edge_list[:, 2], (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
                G = (G + G.T)/2

                rename_solution = np.array(
                    [solution[rename_nodes_list_0[graph_index][i]] for i in range(n)])

                J_sub, h_sub, C_sub = calc_subqubo(
                    [i for i in range(n)], rename_solution, G)
                sol = solve_QAOA(J_sub, h_sub, C_sub, [i for i in range(
                    n)], rename_solution, depth=5, tol=1e-5)
            sub_graph_left_nodes = []
            for index, i in enumerate(sol):
                if i == 0:
                    sub_graph_left_nodes.append(index)
                    solution[rename_nodes_list_0[graph_index][index]] = 0
                else:
                    solution[rename_nodes_list_0[graph_index][index]] = 1
            left_nodes.append(sub_graph_left_nodes)

        ks = (1 << graphs_num)

        result_solution = solution.copy()
        maxx_cut = calc_cut_x(origin_G, solution)
        for k2 in range(1, ks):
            new_solution = solution.copy()
            for graph_index in range(graphs_num):
                if (k2 & (1 << graph_index)) != 0:
                    for i in range(len(rename_nodes_list_0[graph_index])):
                        if new_solution[rename_nodes_list_0[graph_index][i]] == 0:
                            new_solution[rename_nodes_list_0[graph_index][i]] = 1
                        else:
                            new_solution[rename_nodes_list_0[graph_index][i]] = 0
            it_cut = calc_cut_x(origin_G, new_solution)
            if it_cut > maxx_cut:
                result_solution = new_solution.copy()
                maxx_cut = it_cut

        step = 15.0

        # it=0
        while int(step) > 1:
            n_sub = int(step)
            sub_index,J_sub,h_sub,C_sub=self.build_sub_qubo(result_solution,n_sub,origin_G,heads,h=None,C=0)
            new_solution=solve_QAOA(J_sub,h_sub,C_sub,sub_index,result_solution,depth=5,tol=1e-5) # You can change the depth and tolerance of QAOA solver
            cut_temp =calc_cut_x(origin_G, new_solution)
            if cut_temp > maxx_cut:
                result_solution = new_solution.copy()
                maxx_cut = cut_temp

            step *= 0.8
            # else:
            #     break

            # it+=1

        return result_solution, maxx_cut


def rebuild_graph(G):

    G_dict = dict(G.todok())
    g = nx.Graph()
    for node, Jij in G_dict.items():
        if node[0] < node[1]:
            g.add_edge(node[0], node[1])
    return g


def solve(sol, G):
    g = rebuild_graph(G)
    qaoa_in_qaoa = QaoaInQaoa(_iteration=6)     #######
    solution, cut = qaoa_in_qaoa.main(sol, g, G)
    return solution, cut


def run():
    """
    Main run function, for each graph need to run for 20 times to get the mean result.
    Please do not change this function, we use this function to score your algorithm. 
    """
    cut_list = []
    for filename in filelist[:]:
        #print(f'--------- File: {filename}--------')
        g, G = read_graph(filename)
        n = len(g.nodes)  # 图整体规模
        cuts = []
        for turn in range(20):
            #print(f'------turn {turn}------')
            sol = init_solution(n)  # 随机初始化解
            qubo_start = calc_qubo_x(G, sol)
            cut_start = calc_cut_x(G, sol)
            #print('origin qubo:',qubo_start,'|cut:',cut_start)
            solution, cut = solve(sol, G)  # 主要求解函数, 主要代码实现
            cuts.append(cut)
        cut_list.append(cuts)
    return np.array(cut_list)


if __name__ == "__main__":
    # 计算分数
    cut_list = run()
    print(cut_list)
    max_arr = np.array([54, 108, 238, 406, 226, 231])
    size_arr = np.array([40, 80, 40, 80, 80, 80])
    score = np.array(np.mean(cut_list, axis=1))/max_arr*size_arr
    print('score:', np.sum(score))
