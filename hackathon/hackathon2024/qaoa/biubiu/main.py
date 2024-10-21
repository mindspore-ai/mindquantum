import numpy as np
import json


def main(Jc_dict, p, Nq=14):
    '''
        The main function you need to change!!!
    Args:
        Jc_dict (dict): the ising model
        p (int): the depth of qaoa circuit
    Returns:
        gammas (Union[numpy.ndarray, List[float]]): the gamma parameters, the length should be equal to depth p.
        betas (Union[numpy.ndarray, List[float]]): the beta parameters, the length should be equal to depth p.
    '''
    hypergraphs_list = []
    label_list = []
    # k2 0.3=23 0.6=46 0.9=70
    # k3 0.3=89 0.6=178 0.9=268
    # k4 0.3=237 0.6=475 0.9=713
    # k5 0.3=475 0.6=951 0.9=1426
    # 首先判断边数为多少 - 然后压缩
    # 其次判断分布 std uni bimodal
    # 计算均值与标准差，标准差为0，均值为5则为均匀分布，如果均值为0
    edge_value = list(Jc_dict.keys())
    edge_len = len(edge_value)

    max_nodes = 0
    for edge in Jc_dict.keys():
        num_nodes = len(edge)
        if num_nodes > max_nodes:
            max_nodes = num_nodes

    # 判断属于k多少，0.多少，以及是否需要svd
    svd = 0
    prolv = 0
    svdnum = 0
    if max_nodes == 2:
        if edge_len <= 23:
            prolv = 0.3
            svdnum = edge_len
            if edge_len < 23:
                svd = 1
        elif edge_len <= 46:
            prolv = 0.6
            svdnum = edge_len
            if edge_len < 46:
                svd = 1
        elif edge_len <= 70:
            prolv = 0.9
            svdnum = edge_len
            if edge_len < 70:
                svd = 1
        else:
            prolv = 0.9
            # Jc_dict = {k: Jc_dict[k] for k in list(Jc_dict)[:70]}

    if max_nodes == 3:
        if edge_len <= 89:
            prolv = 0.3
            svdnum = edge_len
            if edge_len < 89:
                svd = 1
        elif edge_len <= 178:
            prolv = 0.6
            svdnum = edge_len
            if edge_len < 178:
                svd = 1
        elif edge_len <= 268:
            prolv = 0.9
            svdnum = edge_len
            if edge_len < 268:
                svd = 1
        else:
            prolv = 0.9
            # Jc_dict = {k: Jc_dict[k] for k in list(Jc_dict)[:268]}

    if max_nodes == 4:
        if edge_len <= 237:
            prolv = 0.3
            svdnum = edge_len
            if edge_len < 237:
                svd = 1
        elif edge_len <= 475:
            prolv = 0.6
            svdnum = edge_len
            if edge_len < 475:
                svd = 1
        elif edge_len <= 713:
            prolv = 0.9
            svdnum = edge_len
            if edge_len < 713:
                svd = 1
        else:
            prolv = 0.9
            # Jc_dict = {k: Jc_dict[k] for k in list(Jc_dict)[:713]}

    if max_nodes == 5:
        if edge_len <= 475:
            prolv = 0.3
            svdnum = edge_len
            if edge_len < 475:
                svd = 1
        elif edge_len <= 951:
            prolv = 0.6
            svdnum = edge_len
            if edge_len < 951:
                svd = 1
        elif edge_len <= 1426:
            prolv = 0.9
            svdnum = edge_len
            if edge_len < 1426:
                svd = 1
        else:
            prolv = 0.9
            # Jc_dict = {k: Jc_dict[k] for k in list(Jc_dict)[:1426]}

    if max_nodes < 2:
        svd = 1
    if max_nodes > 5:
        svd = 2

    if svd == 1:
        D = ave_D(Jc_dict, Nq)
        k = order(Jc_dict)
        import csv
        # Read the parameters of infinite size limit, and the maximum order is 6 surported here.
        k = min(k, 6)
        with open('utils/transfer_data.csv', 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if (row[0]) == str(k):
                    new_row = [item for item in row if item != '']
                    length = len(new_row)
                    if length == 3 + 2 * p:
                        gammas1 = np.array([float(new_row[i]) for i in range(3, 3 + p)])
                        betas1 = np.array([float(new_row[i]) for i in range(3 + p, 3 + 2 * p)])
        # rescale the parameters for specific case
        gammas1 = trans_gamma(gammas1, D)
        factor = rescale_factor(Jc_dict)
        gammas1 = gammas1 * factor
        return gammas1, betas1  # or some other default values
    else:
        if svd == 2: # hidden data
            # 判断属于什么分布
            weights = np.array(list(Jc_dict.values()))
            # 计算均值
            mean_weight = np.mean(weights)
            # 计算标准差
            mean_weight = np.std(weights)
            # 判断是否完全相等
            all_equal = np.all(weights == weights[0])
            # 判断是否有大于5的值
            has_greater_than_five = np.any(weights > 5)
            if all_equal == 1:
                fenbu = 'std'
            elif has_greater_than_five == 1:
                fenbu = 'bimodal'
            else:
                fenbu = 'uni'
            X = np.load('./X.npy')
            if p == 4:
                y = np.load('./y_4.npy')
            else:
                y = np.load('./y_8.npy')
            # # 示例: 使用训练好的模型预测新超图的参数
            new_features1 = hypergraph_to_matrix(Jc_dict)
            new_features1 = svd_reduce(new_features1, 10)
            new_features = new_features1.flatten().reshape(1, -1)
            predicted_label = knn(X, y, new_features, k=2, distance_fn=euclidean_distance, choice_fn=mean)
            # print(f"Predicted Parameters: {predicted_label}")
            gammas = np.array(predicted_label[0::2])  # 使用切片，步长为2

            # 偶数位元素（从索引1开始的偶数索引位置）
            betas = np.array(predicted_label[1::2])
            print(gammas, betas)
            return gammas, betas


        else:
            # 判断属于什么分布
            weights = np.array(list(Jc_dict.values()))
            # 计算均值
            mean_weight = np.mean(weights)
            # 计算标准差
            mean_weight = np.std(weights)
            # 判断是否完全相等
            all_equal = np.all(weights == weights[0])
            # 判断是否有大于5的值
            has_greater_than_five = np.any(weights > 5)
            if all_equal == 1:
                fenbu = 'std'
            elif has_greater_than_five == 1:
                fenbu = 'bimodal'
            else:
                fenbu = 'uni'

            for r in range(10):
                Jc_dict1 = load_data(f"data/k{max_nodes}/{fenbu}_p{prolv}_{r}.json")
                # label = load_label(f"utils/label/k{max_nodes}/{fenbu}_p{prolv}_{r}_{p}_labelss.json")
                combined_file_path = f"utils/label/k{max_nodes}/k{max_nodes}.json"
                with open(combined_file_path, 'r') as file:
                    combined_data = json.load(file)
                label = load_label(f"{fenbu}_p{prolv}_{r}_{p}_labelss.json", combined_data)
                matrix1 = hypergraph_to_matrix(Jc_dict1)
                # if svd == 1:
                #     matrix1 = svd_reduce(matrix1, svdnum)
                matrix = matrix1.flatten()
                hypergraphs_list.append(matrix)
                label_list.append(label)
                # print(matrix)

            X = np.array(hypergraphs_list)  # 将矩阵列表转换为一个NumPy数组
            y = np.array(label_list)  # 将标签列表转换为一个NumPy数组

            # # 示例: 使用训练好的模型预测新超图的参数
            new_features1 = hypergraph_to_matrix(Jc_dict)
            # if svd == 1:
            #     new_features1 = svd_reduce(new_features1, svdnum)
            new_features = new_features1.flatten().reshape(1, -1)
            predicted_label = knn(X, y, new_features, k=1, distance_fn=euclidean_distance, choice_fn=mean)
            # print(f"Predicted Parameters: {predicted_label}")
            gammas = np.array(predicted_label[0::2])  # 使用切片，步长为2

            # 偶数位元素（从索引1开始的偶数索引位置）
            betas = np.array(predicted_label[1::2])
            # print(gammas, betas)
            return gammas, betas

def trans_gamma(gammas, D):
    return gammas*np.arctan(1/np.sqrt(D-1))

def rescale_factor(Jc):
    '''
    Get the rescale factor, a technique from arXiv:2305.15201v1
    '''
    import copy
    Jc_dict=copy.deepcopy(Jc)
    keys_len={}
    for key in Jc_dict.keys():
        if len(key) in keys_len:
            keys_len[len(key)]+=1
        else:
            keys_len[len(key)]=1
    norm=0
    for key,val in Jc_dict.items():
        norm+= val**2/keys_len[len(key)]
    norm = np.sqrt(norm)
    return  1/norm

def ave_D(Jc,nq):
    return 2*len(Jc)/nq

def order(Jc):
    return max([len(key)  for key in Jc.keys()])

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    Jc_dict = {}
    for item in range(len(data['c'])):
        Jc_dict[tuple(data['J'][item])] = data['c'][item]
    return Jc_dict

def knn(features, labels, query, k, distance_fn, choice_fn):
    """
    参数:
    features : ndarray，数据点的特征矩阵，每行是一个数据点，每列是一个特征。
    labels : array，数据点的标签数组。
    query : ndarray，需要分类的点。
    k : int，最近邻的数量。
    distance_fn : function，用于计算两点间距离的函数。
    choice_fn : function，用于从最近邻的标签中选择分类标签的函数。
    """
    neighbor_distances_and_indices = []

    # 计算所有点到查询点的距离
    for index, example in enumerate(features):
        # 计算距离
        distance = distance_fn(example, query)
        # 保存距离和索引
        neighbor_distances_and_indices.append((distance, index))

    # 按距离排序并选择最近的k个邻居
    sorted_neighbors = sorted(neighbor_distances_and_indices)
    k_nearest_distances_and_indices = sorted_neighbors[:k]

    # 提取这些邻居的标签
    k_nearest_labels = [labels[i] for distance, i in k_nearest_distances_and_indices]

    # 让这些标签进行投票
    return choice_fn(k_nearest_labels)

def euclidean_distance(point1, point2):
    """
    计算两点间的欧氏距离。
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

def mode(labels):
    """
    返回最常见的标签。
    参数 labels 是一个列表，其中的元素可以是 numpy.ndarray 类型。
    """
    # 将数组转换为元组，以便能够创建集合
    labels = [tuple(label) for label in labels]
    # 计算众数
    return max(set(labels), key=labels.count)

def mean(labels):
    """
    返回标签的平均数。
    参数 labels 是一个列表，其中的元素可以是 numpy.ndarray 类型。
    """
    # 将数组转换为 NumPy 数组以便进行计算
    labels = np.array(labels)
    # 计算平均数
    return np.mean(labels, axis=0)


def svd_reduce(matrix, k):
    """
    使用 SVD 对矩阵进行降维，保留前 k 个奇异值。

    Args:
        matrix (numpy.ndarray): 输入矩阵
        k (int): 保留的奇异值和奇异向量的数量

    Returns:
        reduced_matrix (numpy.ndarray): 降维后的矩阵
    """
    # 计算 SVD
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # 保留前 k 个奇异值和相应的奇异向量
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :k]

    # 计算降维后的矩阵
    reduced_matrix = np.dot(U_k, np.dot(S_k, Vt_k))

    # return reduced_matrix
    return S[:k]

def load_label(filename, combined_data):
    label = combined_data[filename]
    Jc_dict = []
    for value in label.values():
        Jc_dict.append(value)
    return Jc_dict

def hypergraph_to_matrix(hypergraph):
    edges = list(hypergraph.keys())
    weights = list(hypergraph.values())

    num_nodes = 12  # 假设节点编号从0开始
    num_edges = len(edges)

    # 创建点*边的矩阵，初始值为0
    matrix = np.zeros((num_nodes, num_edges))

    for edge_idx, edge in enumerate(edges):
        for node in edge:
            matrix[node, edge_idx] = weights[edge_idx]

    return matrix


