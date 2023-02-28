from graph import Graph, Vertex
import numpy as np


def rule_f(g:Graph):
    # ZX演算过程中，图中的顶点会发生增减
    # 获取最初的所有顶点
    for v1 in list(g.vertices.keys()):
        # 判断当前顶点在化简过程中有没有被删除
        if v1 not in g.vertices.keys():
            continue
        v1 = g.vertices[v1]
        if v1.color == 'red' or v1.color == 'green':    # 不化简黄色顶点
            for v2 in v1.neighbor:
                v2 = g.vertices[v2]
                if v2.color == v1.color:    # 相同颜色
                    v2.phase += v1.phase
                    v2.neighbor.extend(v1.neighbor)    # 合并两个顶点
                    for v3 in v1.neighbor:
                        v3 = g.vertices[v3]
                        v3.neighbor.append(v2.name)
                    g.delete_vertex(v1.name)    # 删除已被合并的顶点
                    g.clear()    # 清除过程中可能产生的环



def rule_id(g:Graph):
    for v1 in list(g.vertices.keys()):
        if v1 not in g.vertices.keys():
            continue
        v1 = g.vertices[v1]
        if v1.phase == 0 or list(v1.phase.values()) == [0.0]*len(list(v1.phase.values())):    # phase =0（phase可能是列表）
            flag = True
            for v2 in v1.neighbor:
                v2 = g.vertices[v2]
                if v2.qubit != v1.qubit:    # 与其他量子位上的顶点相关
                    flag = False
            if flag:
                for v2 in v1.neighbor:
                    v2 = g.vertices[v2]
                    v2.neighbor.extend(v1.neighbor)    # 将前一个顶点与后一个顶点相连，略过当前顶点
                g.delete_vertex(v1.name)
                g.clear()



def rule_hh(g:Graph):
    for v1 in list(g.vertices.keys()):
        if v1 not in g.vertices.keys():
            continue
        v1 = g.vertices[v1]
        if v1.color == 'yellow':
            for v2 in v1.neighbor:
                v2 = g.vertices[v2]
                if v2.qubit != v1.qubit:    # 当前顶点与其他qubit上的顶点相关
                    return
            for v2 in v1.neighbor:
                v2 = g.vertices[v2]
                if v2.color == 'yellow':    # 都是yellow顶点
                    v1.neighbor.remove(v2.name)    # 最左边和最右边相连，略过两个yellow顶点
                    v2.neighbor.remove(v1.name)
                    for v3 in v1.neighbor:
                        v3 = g.vertices[v3]
                        v3.neighbor.extend(v2.neighbor)
                    for v4 in v2.neighbor:
                        v4 = g.vertices[v4]
                        v4.neighbor.extend(v1.neighbor)
                    g.delete_vertex(v1.name)
                    g.delete_vertex(v2.name)
                    g.clear()



def rule_h(g:Graph):
    for v1 in list(g.vertices.keys()):
        if v1 not in g.vertices.keys():
            continue
        v1 = g.vertices[v1]
        if v1.color == 'green':
            flag = True
            for v2 in v1.neighbor:
                v2 = g.vertices[v2]
                if v2.color != 'yellow':    # 不满足所有邻居都是yellow
                    flag = False
                    break
            if flag:
                v1.color = 'red'    # 当前顶点从green变成red
                for v2 in list(v1.neighbor):    # 顶点的所有邻居都是yellow，略过并删除这些yellow顶点
                    v2 = g.vertices[v2]
                    v2.neighbor.remove(v1.name)
                    v1.neighbor.extend(v2.neighbor)
                    for v3 in v2.neighbor:
                        v3 = g.vertices[v3]
                        v3.neighbor.append(v1.name)
                    g.delete_vertex(v2.name)
                    g.clear()



def rule_b(g:Graph):
    for v1 in list(g.vertices.keys()):
        if v1 not in g.vertices.keys():
            continue
        v1 = g.vertices[v1]
        if v1.color == 'green':    # 沙漏左上角
            for v2 in v1.neighbor:
                v2 = g.vertices[v2]
                if v2.color == 'red' and v2.qubit == v1.qubit:    # 沙漏右上角
                    for v3 in v2.neighbor:
                        v3 = g.vertices[v3]
                        if v3.color == 'green' and v3.qubit != v2.qubit:    # 沙漏左下角
                            for v4 in v3.neighbor:
                                v4 = g.vertices[v4]
                                if v4.color == 'red' and v4.qubit == v3.qubit and v1.name in v4.neighbor:    # 沙漏右下角
                                    v1.color = 'red'    # 反转颜色，并将沙漏左侧合二为一
                                    v3.neighbor.remove(v2.name)
                                    v3.neighbor.remove(v4.name)
                                    v1.neighbor.extend(v3.neighbor)
                                    v2.color = 'green'    # 反转颜色，并将沙漏右侧合二为一
                                    v4.neighbor.remove(v1.name)
                                    v4.neighbor.remove(v3.name)
                                    v2.neighbor.extend(v4.neighbor)
                                    for v in v3.neighbor:
                                        v = g.vertices[v]
                                        v.neighbor.append(v1.name)
                                    for v in v4.neighbor:
                                        v = g.vertices[v]
                                        v.neighbor.append(v2.name)
                                    g.delete_vertex(v3.name)
                                    g.delete_vertex(v4.name)
                                    g.clear()



def rule_1(g:Graph):
    for v1 in list(g.vertices.keys()):
        if v1 not in g.vertices.keys():
            continue
        v1 = g.vertices[v1]
        if v1.color == 'green':
            for v2 in v1.neighbor:
                v2 = g.vertices[v2]
                if v2.color == 'red' and v2.neighbor.count(v1.name) == 2:    # 红绿顶点，且两顶点间有两条边
                    while v2.name in g.vertices[v1.name].neighbor:    # 删除相连的边
                        v1.neighbor.remove(v2.name)
                    while v1.name in g.vertices[v2.name].neighbor:
                        v2.neighbor.remove(v1.name)



def rule_2(g:Graph):
    for v1 in list(g.vertices.keys()):
        if v1 not in g.vertices.keys():
            continue
        v1 = g.vertices[v1]
        if v1.color== 'red':    # 上一
            for v2 in v1.neighbor:
                v2 = g.vertices[v2]
                if v2.color == 'green' and v2.qubit == v1.qubit:    # 上二
                    for v3 in v2.neighbor:
                        v3 = g.vertices[v3]
                        if v3.color == 'red' and v3.qubit == v2.qubit and v3.name != v1.name:    # 上三
                            for v6 in v3.neighbor:
                                v6 = g.vertices[v6]
                                if v6.color == 'green' and v6.qubit != v3.qubit:    # 下三
                                    for v5 in v6.neighbor:
                                        v5 = g.vertices[v5]
                                        if v5.color == 'red' and v5.qubit == v6.qubit and v2.name in v5.neighbor:    # 下二
                                            for v4 in v5.neighbor:
                                                v4 = g.vertices[v4]
                                                if v4.color == 'green' and v4.qubit == v5.qubit and v1.name in v4.neighbor:    # 下一
                                                    g.delete_vertex(v2.name)    # 全部删除
                                                    g.delete_vertex(v5.name)
                                                    v1.neighbor.remove(v4.name)
                                                    v4.neighbor.remove(v1.name)
                                                    v3.neighbor.remove(v6.name)
                                                    v6.neighbor.remove(v3.name)
                                                    for v in v1.neighbor:
                                                        v = g.vertices[v]
                                                        v.neighbor.extend(v6.neighbor)
                                                    for v in v6.neighbor:
                                                        v = g.vertices[v]
                                                        v.neighbor.extend(v1.neighbor)
                                                    for v in v4.neighbor:
                                                        v = g.vertices[v]
                                                        v.neighbor.extend(v3.neighbor)
                                                    for v in v3.neighbor:
                                                        v = g.vertices[v]
                                                        v.neighbor.extend(v4.neighbor)
                                                    g.delete_vertex(v1.name)
                                                    g.delete_vertex(v3.name)
                                                    g.delete_vertex(v4.name)
                                                    g.delete_vertex(v6.name)
                                                    g.clear()
                                                    for v in g.vertices.values():    # 化简后只剩交叉
                                                        if v.qubit == v3.qubit and v.name > v3.name:
                                                            v.qubit = v6.qubit
                                                        elif v.qubit == v6.qubit and v.name > v6.name:
                                                            v.qubit = v3.qubit



def rule_neg(g:Graph):
    for v1 in list(g.vertices.keys()):
        if v1 not in g.vertices.keys():
            continue
        v1 = g.vertices[v1]
        if v1.phase != 0:
            for v2 in v1.neighbor:
                v2 = g.vertices[v2]
                if v2.color == v1.color and v2.phase == -v1.phase:    # 相同颜色、相反相位
                    v2.phase = 0
                    v2.neighbor.extend(v1.neighbor)
                    for v3 in v1.neighbor:
                        v3 = g.vertices[v3]
                        v3.neighbor.append(v2.name)
                    g.delete_vertex(v1.name)
                    g.clear()
                    rule_id(g)



# 利用以上规则进行化简
def simplify(g : Graph):
    temp = []
    while(temp != list(g.vertices.keys())):
        temp = list(g.vertices.keys())    # temp用于对比本轮循环是否有效化简
        rule_hh(g)
        rule_h(g)
        rule_neg(g)
        rule_b(g)
        rule_2(g)
        rule_f(g)
        rule_1(g)
        rule_id(g)


