# 顶点
class Vertex:
    def __init__(self, name, color,qubit, neighbor, phase=0.0):
        self.name = name    # 量子门序号
        self.color = color
        self.phase = phase   # 量子门相位
        self.qubit = qubit     # 作用在哪个量子比特上
        self.neighbor = neighbor    # 前一个量子门、后一个量子门、存在控制关系的量子门的序号



# ZX图默认无环，省略了ZX演算的无环规则（本文中，所有“无环”均指不存在单条边构成的环）
class Graph:
    def __init__(self):
        self.vertices = {}    # 初始图，空
        self.count = 0       # 顶点总数，只增不减，也用于给新顶点命名

    # 新增边
    def add_edge(self, from_vertex, to_vertex):
        self.vertices[from_vertex].neighbor.append(to_vertex)

    # 新增顶点
    def add_vertex(self, color, qubit, neighbor, phase=0.0):
        name = self.count
        self.count += 1
        self.vertices[name] = Vertex(name, color, qubit, neighbor, phase)    # 新增顶点时已在图中绘制了与相邻顶点的单向边
        for v in neighbor:    # 再绘制反方向的边
            self.add_edge(v, name)

    # 打印图信息
    def print(self):
        print("==================graph message==================")
        for v in self.vertices.values():
            print(v.name, '\t', v.neighbor, '\t', v.color, '\t', v.phase)
        print('\n')

    # 删除自身的环
    def clear(self):
        for v in self.vertices.values():
            while v.name in v.neighbor:
                self.vertices[v.name].neighbor.remove(v.name)

    # 删除顶点
    def delete_vertex(self, name):
        for v in self.vertices.values():
            while name in v.neighbor:    # 删除与该顶点有关的所有边（此处删除终点是该顶点的边）
                self.vertices[v.name].neighbor.remove(name)
        self.vertices.pop(name)    # 删除该顶点（此处删除起点是该顶点的边）

    # 两个电路是否等价
    def equiv(self):
        if len(self.vertices) == 0:    # 等价的两个电路，经过ZX化简后，图中无顶点
            print("Equivalent!")
        else:
            print("Not sure!")



# 将量子线路绘制成ZX图
def draw_graph(circ):
    g = Graph()
    # last_name保存每个qubit上当前的最后一个顶点的name
    last_name = [-1] * circ.n_qubits
    for gate in circ:
        if gate.name == 'H':
            # 当前qubit上已有顶点
            if last_name[gate.obj_qubits[0]] != -1:
                g.add_vertex('yellow', gate.obj_qubits[0], [last_name[gate.obj_qubits[0]]])
            # 当前qubit上暂无顶点
            else:
                g.add_vertex('yellow', gate.obj_qubits[0], [])
            # 更新当前qubit上最后一个顶点的name为新增顶点
            last_name[gate.obj_qubits[0]] = g.count-1
        if gate.name == 'RX':
            if last_name[gate.obj_qubits[0]] != -1:
                # 顶点phase = RX门参数
                g.add_vertex('red', gate.obj_qubits[0], [last_name[gate.obj_qubits[0]]], gate.coeff)
            else:
                g.add_vertex('red', gate.obj_qubits[0], [], gate.coeff)
            last_name[gate.obj_qubits[0]] = g.count-1
        if gate.name == 'RZ':
            if last_name[gate.obj_qubits[0]] != -1:
                g.add_vertex('green', gate.obj_qubits[0], [last_name[gate.obj_qubits[0]]], gate.coeff)
            else:
                g.add_vertex('green', gate.obj_qubits[0], [], gate.coeff)
            last_name[gate.obj_qubits[0]] = g.count-1
        if gate.name == 'CNOT':
            # 控制位顶点
            if last_name[gate.obj_qubits[1]] != -1:
                g.add_vertex('green', gate.obj_qubits[1], [last_name[gate.obj_qubits[1]]])
            else:
                g.add_vertex('green', gate.obj_qubits[1], [])
            last_name[gate.obj_qubits[1]] = g.count-1
            # 受控位顶点
            if last_name[gate.obj_qubits[0]] != -1:
                # 受控顶点neighbor中默认包含控制位顶点
                g.add_vertex('red', gate.obj_qubits[0], [last_name[gate.obj_qubits[0]],  g.count-1])
            else:
                g.add_vertex('red', gate.obj_qubits[0], [g.count-1])
            last_name[gate.obj_qubits[0]] = g.count-1
    return g




