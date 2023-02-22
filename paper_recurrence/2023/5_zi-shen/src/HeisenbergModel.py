from mindquantum import Hamiltonian, QubitOperator
import networkx as nx


class HeisenbergModel:

    def __init__(self, J, h_z, graph: nx.Graph):
        """
        Constructor of HeisenbergModel.
        :param J: represents interaction strength between particles;
        :param h_z: represents the strength of the external magnetic field (along z direction);
        :param graph: the graph to generate Heisenberg model.
        """
        self.J = J
        self.h_z = h_z
        self.graph = graph
        self.n_qubits = graph.number_of_nodes()

    def ham(self) -> Hamiltonian:
        """
        Generate a Heisenberg Hamiltonian based on self.graph.
        :return: a Hamiltonian
        """
        ham = QubitOperator()
        for i in self.graph.edges:
            ham += self.J * QubitOperator(f'X{i[0]} X{i[1]}')
            ham += self.J * QubitOperator(f'Y{i[0]} Y{i[1]}')
            ham += self.J * QubitOperator(f'Z{i[0]} Z{i[1]}')
        for i in self.graph.nodes:
            ham += self.h_z * QubitOperator(f'Z{i}')
        return Hamiltonian(ham)

    def draw_graph(self) -> None:
        nx.draw(self.graph)


if __name__ == '__main__':
    graph_1d = nx.random_graphs.random_regular_graph(2, 6)
    print('Heisenberg Hamiltonian = ', HeisenbergModel(1, 1, graph_1d).ham())
