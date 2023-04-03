"""Quantum device module"""
from __future__ import annotations
import mindquantum.mqbackend.device
import typing
import mindquantum.mqbackend.gate
from .sabre import SABRE

__all__ = [
    "GridQubits",
    "LinearQubits",
    "QubitNode",
    "QubitsTopology",
    "SABRE",
]

class QubitsTopology:
    """
    Topology of qubit in physical device.

    Topology is construct by different QubitNode, and you can set the property of each qubit node directly.

    Args:
        List[QubitNode], all qubit nodes in this topology.

    Examples:
        >>> from mindquantum.device import QubitsTopology
        >>> from mindquantum.device import QubitNode
        >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
        >>> topology[0] >> topology[1]
        >>> topology.is_coupled_with(0, 1)
        True
        >>> topology.set_color(0, "#121212")
        >>> topology[0].color
        '#121212'
    """

    def __getitem__(self, id: int) -> QubitNode:
        """
        Get qubit node base on qubit id.

        Args:
            id (int), the identity number of qubit you want.

        Returns:
            QubitNode, the qubit node with given id.

        Examples:
            >>> from mindquantum.device import QubitsTopology
            >>> from mindquantum.device import QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology[2].id
            2
        """
    def __init__(self, qubits: typing.List[QubitNode]) -> None:
        """Initialize a physical qubit topology."""
    def add_qubit_node(self, qubit: QubitNode) -> None:
        """
        Add a qubit node into this topology.

        Args:
            qubit (QubitNode): the qubit you want to add into this topology.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(2)])
            >>> topology.add_qubit_node(QubitNode(2));
            >>> topology.all_qubit_id()
            {0, 1, 2}
        """
    def all_qubit_id(self) -> typing.Set[int]:
        """
        Get all qubit id.

        Returns:
            Set[int], all qubit id in this qubit topology.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(2)])
            >>> topology.add_qubit_node(QubitNode(2));
            >>> topology.all_qubit_id()
            {0, 1, 2}
        """
    def dict(self) -> typing.Dict[int, QubitNode]:
        """
        Get the map of qubits with key as qubit id and value as qubit itself.

        Returns:
            Dict[int, QubitNode], a dict with qubit id as key and qubit as value.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(2)])
            >>> topology.dict()
            {1: <mindquantum.mqbackend.device.QubitNode at 0x7fee5e124a70>,
            0: <mindquantum.mqbackend.device.QubitNode at 0x7feef3998270>}
        """
    def edges_with_id(self) -> typing.Set[typing.Tuple[int, int]]:
        """
        Get edges with id of two connected qubits.

        Returns:
            Set[Tuple[int, int]], all connected edges in this qubit topology.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(2)])
            >>> topology[0] << topology[1]
            >>> topology.edges_with_id()
            {(0, 1)}
        """
    def edges_with_poi(self) -> typing.Set[typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]]:
        """
        Get edges with position of two connected qubits.

        Returns:
            Set[Tuple[Tuple[float, float], Tuple[float, float]]], the x and y position of two connected qubits.

        Examples:
            >>> from mindquantum.device import QubitNode, QubitsTopology
            >>> q0 = QubitNode(0, poi_x=0, poi_y=0)
            >>> q1 = QubitNode(1, poi_x=1, poi_y=0)
            >>> q0 >> q1
            >>> topology = QubitsTopology([q0, q1])
            >>> topology.edges_with_poi()
            {((0.0, 0.0), (1.0, 0.0))}
        """
    def has_qubit_node(self, id: int) -> bool:
        """
        Check whether a qubit is in this topology.

        Args:
            id (int), the id of qubit you want to check.

        Returns:
            bool, whether this topology has qubit node with given id.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(2)])
            >>> topology.has_qubit_node(0)
            True
        """
    def is_coupled_with(self, id1: int, id2: int) -> bool:
        """
        Check whether two qubit nodes are coupled.

        Args:
            id1 (int): the id of one qubit you want to check.
            id2 (int): the id of the other qubit you want to check.

        Returns:
            bool, whether two qubits node with given ids coupled.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(2)])
            >>> topology.is_coupled_with(0, 1)
            False
            >>> topology[0] >> topology[1]
            >>> topology.is_coupled_with(0, 1)
            True
        """
    def isolate_with_near(self, id: int) -> None:
        """
        Disconnect with all coupling qubits.

        Args:
            id (int): the id of qubit you want to disconnect with all nearby qubits.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology[0] >> topology[1] >> topology[2]
            >>> topology.edges_with_id()
            {(0, 1), (1, 2)}
            >>> topology.isolate_with_near(1)
            >>> topology.edges_with_id()
            set()
        """
    def n_edges(self) -> int:
        """
        Get total connected edge number.

        Returns:
            int, the edge number of this qubit topology.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology[0] >> topology[1] >> topology[2]
            >>> topology.n_edges()
            2
        """
    def choose(self, ids: typing.List[int]) -> typing.List[QubitNode]:
        """
        Choose qubit nodes based on given qubit id.

        Args:
            ids (List[int]): A list of qubit node id.

        Returns:
            List[QubitNode], a list of qubit node based on given qubit node id.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology[0] >> topology[1] >> topology[2]
            >>> nodes = topology.choose([0, 1])
            >>> print(nodes[0].id, nodes[1].id)
            0 1
        """
    def remove_isolate_node(self) -> None:
        """
        Remove qubit node that do not connect with any other qubits.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology[0] >> topology[1] >> topology[2]
            >>> topology.edges_with_id()
            {(0, 1), (1, 2)}
            >>> topology.isolate_with_near(1)
            >>> topology.all_qubit_id()
            {0, 1, 2}
            >>> topology.remove_isolate_node()
            >>> topology.all_qubit_id()
            set()
        """
    def remove_qubit_node(self, id: int) -> None:
        """
        Remove a qubit node out of this topology.

        Args:
            id (int): the id of qubit you want to remove.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology.remove_qubit_node(1)
            >>> topology.all_qubit_id()
            {0, 2}
        """
    @typing.overload
    def set_color(self, id: int, color: str) -> None:
        """
        Set color of certain qubit.

        Args:
            id (int), the id of qubit you want to change color.
            color (str), the new color.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology.set_color(0, "#ababab")
            >>> topology[0].color
            '#ababab'
        """
    @typing.overload
    def set_color(self, color_map: typing.Dict[int, str]) -> None:
        """
        Set color of many qubits.

        Args:
            color_map (Dict[int, str]): a map with qubit id as key and new color as value.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology.set_color({0: '#000000', 1: '#111111', 2: '#222222'})
            >>> topology[2].color
            '#222222'
        """
    @typing.overload
    def set_position(self, id: int, poi_x: float, poi_y: float) -> None:
        """
        Set position of a certain qubit.

        Args:
            id (int): the id of qubit you want to change position.
            poi_x (float): new x position.
            poi_y (float): new y position.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology.set_position(0, 1, 1)
            >>> topology[0].poi_x, topology[0].poi_y
            (1.0, 1.0)
        """
    @typing.overload
    def set_position(self, poi_map: typing.Dict[int, typing.Tuple[float, float]]) -> None:
        """
        Set position of many qubits.

        Args:
            poi_map (Dict[int, Tuple[float, float]]), the position map with qubit id as key and position as value.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology.set_position({0: (0, 0), 1: (1, 0), 2: (2, 0)})
            >>> topology[2].poi_x, topology[2].poi_y
            (2.0, 0.0)
        """
    def size(self) -> int:
        """
        Get total qubit number.

        Returns:
            int, the total qubit number.

        Examples:
            >>> from mindquantum.device import QubitsTopology, QubitNode
            >>> topology = QubitsTopology([QubitNode(i) for i in range(3)])
            >>> topology.size()
            3
        """
    pass

class LinearQubits(QubitsTopology):
    """
    Linear qubit topology.

    Args:
        n_qubits (int), total qubit size.

    Examples:
        >>> from mindquantum.device import LinearQubits
        >>> topology = LinearQubits(5)
        >>> topology.is_coupled_with(0, 1)
        True
        >>> topology.is_coupled_with(0, 2)
        False
    """

    def __init__(self, n_qubits: int) -> None:
        """Initialize a linear qubit topology."""
    pass

class QubitNode:
    """
    Qubit node.

    A qubit node has a id, a position and a color (if you want to draw it). You can connect two qubits with
    '>>' or '<<' and disconnect two qubits with '<' or '>'. For more detail, please see :class:`QubitNode.__rshift__`,
    :class:`QubitNode.__lshift__`, :class:`QubitNode.__lt__` and :class:`QubitNode.__gt__`.

    Args:
        id (int): the identity number of this qubit.
        color (str): the face color of this qubit.
        poi_x (float): x position in canvas.
        poi_y (float): y position in canvas.

    Examples:
        >>> from mindquantum.device import QubitNode
        >>> q0 = QubitNode(0)
        >>> q1 = QubitNode(1)
        >>> q = q0 << q1
        >>> q0.id == q.id
        True
    """

    def __gt__(self, other: QubitNode) -> QubitNode:
        """
        Disconnect with other qubit node and return rhs.

        Args:
            other (QubitNode): qubit node you want to disconnect with.

        Returns:
            QubitNode, the right hand size qubit.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> from mindquantum.device import QubitsTopology
            >>> q0 = QubitNode(0)
            >>> q1 = QubitNode(1)
            >>> topology = QubitsTopology([q0, q1])
            >>> q0 << q1
            >>> topology.is_coupled_with(0, 1)
            True
            >>> q0 > q1
            >>> topology.is_coupled_with(0, 1)
            False
        """
    def __init__(self, id: int, color: str = '#000000', poi_x: float = 0.0, poi_y: float = 0.0) -> None:
        """Initialize a qubit node."""
    def __int__(self) -> int:
        """
        Convert the qubit to int.

        Returns:
            int, the id of qubit.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> q0 = QubitNode(0)
            >>> int(q0)
            0
        """
    def __lshift__(self, other: QubitNode) -> QubitNode:
        """
        Connect with other qubit node and return lhs.

        Args:
            other (QubitNode): qubit node you want to connect with.

        Returns:
            QubitNode, the left hand size qubit.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> from mindquantum.device import QubitsTopology
            >>> q0 = QubitNode(0)
            >>> q1 = QubitNode(1)
            >>> topology = QubitsTopology([q0, q1])
            >>> q = q0 << q1
            >>> q.id == q0.id
            True
            >>> topology.is_coupled_with(0, 1)
            True
        """
    def __lt__(self, other: QubitNode) -> QubitNode:
        """
        Disconnect with other qubit node and return lhs.

        Args:
            other (QubitNode): qubit node you want to connect with.

        Returns:
            QubitNode, the left hand size qubit.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> from mindquantum.device import QubitsTopology
            >>> q0 = QubitNode(0)
            >>> q1 = QubitNode(1)
            >>> topology = QubitsTopology([q0, q1])
            >>> q0 << q1
            >>> topology.is_coupled_with(0, 1)
            True
            >>> q = q0 < q1
            >>> topology.is_coupled_with(0, 1)
            False
            >>> q.id == q0.id
            True
        """
    def __rshift__(self, other: QubitNode) -> QubitNode:
        """
        Connect with other qubit node and return rhs.

        Args:
            other (QubitNode): qubit node you want to connect with.

        Returns:
            QubitNode, the right hand side qubit.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> from mindquantum.device import QubitsTopology
            >>> q0 = QubitNode(0)
            >>> q1 = QubitNode(1)
            >>> topology = QubitsTopology([q0, q1])
            >>> q = q0 >> q1
            >>> q.id == q1.id
            True
            >>> topology.is_coupled_with(0, 1)
            True
        """
    def set_poi(self, poi_x: float, poi_y: float) -> None:
        """
        Set the position of this qubit.

        Args:
            poi_x (float): x position of qubit in canvas.
            poi_y (float): y position of qubit in canvas.

        Examples:
            >>> from mindquantum.device import QubitNode
            >>> q0 = QubitNode(1, poi_x=0, poi_y=1)
            >>> q0.set_poi(1, 0)
            >>> print(q0.poi_x, q0.poi_y)
            1.0 0.0
        """
    @property
    def color(self) -> str:
        """
        Get the color of this qubit.

        Returns:
            str, the color of qubit node.
        """
    @property
    def id(self) -> int:
        """
        Get the id of this qubit.

        Returns:
            int, the id of this qubit.
        """
    @property
    def poi_x(self) -> float:
        """
        X position of this qubit.

        Returns:
            float, the x position.
        """
    @property
    def poi_y(self) -> float:
        """
        Y position of this qubit.

        Returns:
            float, the y position.
        """
    pass

class GridQubits(QubitsTopology):
    """
    Grid qubit topology.

    Args:
        n_row (int): how many rows of your grid qubits.
        n_col (int): how many columns of your grid quits.

    Examples:
        >>> from mindquantum.device import GridQubits
        >>> topology = GridQubits(2, 3)
        >>> topology.n_row()
        2
    """

    def __init__(self, n_row: int, n_col: int) -> None:
        """Initialize a grid topology with row and col number."""
    def n_col(self) -> int:
        """
        Get column number.

        Returns:
            int, the column number.

        Examples:
            >>> from mindquantum.device import GridQubits
            >>> topology = GridQubits(2, 3)
            >>> topology.n_col()
            3
        """
    def n_row(self) -> int:
        """
        Get row number.

        Examples:
            >>> from mindquantum.device import GridQubits
            >>> topology = GridQubits(2, 3)
            >>> topology.n_row()
            2
        """
    pass
