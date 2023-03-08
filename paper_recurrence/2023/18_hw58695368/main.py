from src.rqaoa import rqaoa                                   # RQAOA
from src.rqaoa import rqaoa_recursion, rqaoa_translate        # RQAOA 分解
from src.utils import generate_graph                          # 测试图结构
from src.utils import maxcut_enum                             # 枚举法求解MaxCut
from src.qaoa import qaoa, get_partition, get_expectation     # QAOA
import os
import pickle
import networkx as nx

def main():
    """Main."""
    nc_list = [8, 6]
    for i in range(10):
        print('problem instance: ', i)
        path = f'./demo/demo{i}'
        if not os.path.exists(path):
            os.mkdir(path)
        path_name = f'{path}/demo{i}_n20'
        graph, C_max = generate_graph_n20()                     # 生成图
        g = graph
        C_qaoa = get_qaoa_expect(g)                             # QAOA结果
        save(f'{path_name}.pkl', {'g':g, 'Xi':[]})              # 保存图
        Xi = []
        C_rqaoa = dict()
        for nc in nc_list:
            g, Xi = rqaoa_recursion(g, nc, Xi, iter_show=0)     # 消元到nc节点
            save(f'{path_name}_to_n{len(g.nodes)}.pkl', {'g':g, 'Xi':Xi})
            C_rqaoa[nc] = get_rqaoa_result(graph, Xi)           # RQAOA结果
        save(f'{path_name}_result.pkl', {'C_max':C_max,
                                        'C_qaoa':C_qaoa,
                                        'C_rqaoa':C_rqaoa})     # 保存最终结果

def generate_graph_n20():
    """Generate 20-nodes graph structure."""
    C_max = 0
    while C_max < 5:
        g = generate_graph(20)
        C_max, _ = maxcut_enum(g, 'J')
    return g, C_max
def save(filepath, data):
    """Save data."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
def load(filepath):
    """Load data."""
    pass
def get_qaoa_expect(g):
    """Get the expection of QAOA."""
    circ, pr = qaoa(g, 'J')
    expect = get_expectation(g, circ, pr, 'J')
    J = 0
    for i in g.edges:
        J += g.edges[i]['J']
    C_qaoa_expect = (J - expect.real) / 2
    return C_qaoa_expect
def get_rqaoa_result(g, Xi):
    """Get the result of RQAOA."""
    res = rqaoa_translate(g, Xi, False)
    l = []
    for k in res:
        if res[k] == -1:
            l.append(k)
    return nx.cut_size(g, l, weight='J')

if __name__ == '__main__':
    main()
    print('finished!')
