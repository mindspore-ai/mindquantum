import numpy as np
import math
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

class Graph():
    '''
    A graph is saved in both an adjoint matrix and edge list.
    '''
    def __init__(self, v:list=None, edges:list=None,adjoint=None) -> None:

        self.v = v
        self.n_v = len(v)
        self.e = edges
        self.adj = adjoint

        if self.adj is None:
            self._edges_to_adjoint()

        if self.e is None:
            self._adjoint_to_edges()

        self.v2i = {v[i]:i for i in range(self.n_v)}

    def _edges_to_adjoint(self) -> None:
        self.adj = np.zeros((self.n_v, self.n_v))
        for edge in self.e:
            v1 = edge[0]
            v2 = edge[1]
            w = edge[2]
            self.adj[v1][v2] = w
            self.adj[v2][v1] = w

    def _adjoint_to_edges(self) -> None:
        self.e = []

        for i in range(self.n_v):
            for j in range(i+1, self.n_v):
                if self.adj[i][j] != 0:
                    self.e.append((i, j, self.adj[i][j].item()))

    def graph_partition(self, n:int, policy:str='random',n_sub=1) -> list:
        '''
        n : Allowable qubit number.

        policy : Partition strategy. Default is 'random'. Another is 'modularity', which partitions graph basing on greedy modularity method.

        n_sub : number of subgraphs. Only use in 'modularity'.
        ''' 
        H = []
        v = self.v

        if policy == 'modularity':
            G = nx.Graph()
            G.add_nodes_from(v)
            for x in self.e:
                G.add_edge(x[0],x[1],weight=x[2])
            c = greedy_modularity_communities(G,n_communities=n_sub)
            sub_list = [list(x) for x in c]
            for x in sub_list:
                if len(x) > n:
                    n_ssub = math.ceil(len(x) / n)
                    
                    ssub_list = [x[n*i:n*(i+1)] for i in range(n_ssub)]
                    for i in range(n_ssub):
                        A = self.adj[ssub_list[i]][:,ssub_list[i]]
                        H.append(Graph(v=ssub_list[i], adjoint=A))
                else:
                    A = self.adj[x][:,x]
                    H.append(Graph(v=x, adjoint=A))
        if policy == 'random':
            n_sub = math.ceil(self.n_v / n)
            np.random.shuffle(v)
            sub_list = [v[n*i:n*(i+1)] for i in range(n_sub)]
            for i in range(n_sub):
                A = self.adj[sub_list[i]][:,sub_list[i]]
                H.append(Graph(v=sub_list[i], adjoint=A))
        return H