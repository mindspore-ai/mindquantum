# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
from networkx import Graph

class MyGraph(Graph):
    def add_edge_as(self, u_of_edge, v_of_edge, **attr):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Edge attributes can be specified with keywords or by directly
        accessing the edge's attribute dictionary. See examples below.

        If the edge already exists, its attributes will be accumulated.

        Args:
            u_of_edge, v_of_edge (nodes): 
                Nodes can be, for example, strings or numbers.
                Nodes must be hashable (and not None) Python objects.
            attr (keyword arguments, optional): 
                Edge data (or labels or objects) can be assigned using
                keyword arguments.

        Examples:
            The following all add the edge e=(1, 2) to graph G:

            >>> G = Graph()
            >>> e = (1, 2)
            >>> G.add_edge_as(1, 2)  # explicit two-node form
            >>> G.add_edge_as(*e)  # single edge as tuple of two nodes

            Associate data to edges using keywords:

            >>> G.add_edge_as(1, 2, weight=3)
            >>> G.add_edge_as(1, 3, weight=7, capacity=15, length=342.7)
        """
        u, v = u_of_edge, v_of_edge
        # add nodes
        if u not in self._node:
            if u is None:
                raise ValueError("None cannot be a node")
            self._adj[u] = self.adjlist_inner_dict_factory()
            self._node[u] = self.node_attr_dict_factory()
        if v not in self._node:
            if v is None:
                raise ValueError("None cannot be a node")
            self._adj[v] = self.adjlist_inner_dict_factory()
            self._node[v] = self.node_attr_dict_factory()
        # add the edge
        datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
        # attribute superposition
        for k, vu in attr.items():
            datadict[k] = datadict.get(k, 0) + vu
        self._adj[u][v] = datadict
        self._adj[v][u] = datadict
    def clean_edge(self, key, th=0):
        """
        Clean the edge where the keyword argument tends to zero.

        Args:
            key (str): The key of keyword argument of edge data.
            th (float): Threshold.

        Examples:
            >>> G = Graph()
            >>> G.add_edge_as(1, 2, weight=3)
            >>> G.add_edge_as(1, 2, weight=-3)
            >>> G.clean_edge('weight', 1e-6)
        """
        for i in self.edges:
            if abs(self.edges[i][key]) <= th:
                self.remove_edge(*i)
