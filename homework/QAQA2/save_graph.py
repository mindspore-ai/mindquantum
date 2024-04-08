import json
import networkx as nx


d = 9
node = 60
g = nx.random_regular_graph(d,node)
n_v = nx.number_of_nodes(g)
n_e = nx.number_of_edges(g)
edges = []
for e in g.edges():
    edges.append([e[0],e[1],1])
test = {"n_v": n_v, "d":d, "type":"regular", "IsWeighted":0, "n_e": n_e, "edges": edges}
#转化为json格式文件
test_json = json.dumps(test) 

#将json文件保存为.json格式文件
with open('./data/test.json','w+') as file:
    file.write(test_json)

#读取.json格式文件的内容
with open('./data/test.json','r+') as file:
    content=file.read()
content=json.loads(content)

