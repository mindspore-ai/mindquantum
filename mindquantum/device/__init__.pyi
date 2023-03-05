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
    """

    def __getitem__(self, arg0: int) -> QubitNode:
        """
        Get qubit node base on qubit id.
        """
    def __init__(self, arg0: typing.List[QubitNode]) -> None:
        """
        Initialize a physical qubit topology.
        """
    def add_qubit_node(self, arg0: QubitNode) -> None:
        """
        Add a qubit node into this topology.
        """
    def all_qubit_id(self) -> typing.Set[int]:
        """
        Get total qubit id.
        """
    def dict(self) -> typing.Dict[int, QubitNode]:
        """
        Get the map of qubits with key as qubit id and value as qubit itself.
        """
    def edges_with_id(self) -> typing.Set[typing.Tuple[int, int]]:
        """
        Get edges with id of two connected qubits.
        """
    def edges_with_poi(self) -> typing.Set[typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]]:
        """
        Get edges with position of two connected qubits.
        """
    def has_qubit_node(self, arg0: int) -> bool:
        """
        Check whether a qubit is in this topology.
        """
    def is_coupled_with(self, id1: int, id2: int) -> bool:
        """
        Check whether two qubit nodes are coupled.
        """
    def isolate_with_near(self, id: int) -> None:
        """
        Disconnect with all coupling qubits.
        """
    def n_edges(self) -> int:
        """
        Get total connected edge number.
        """
    def remove_isoloate_node(self) -> None:
        """
        Remove qubit node that do not connect with any other qubits.
        """
    def remove_qubit_node(self, arg0: int) -> None:
        """
        Remove a qubit node out of this topology.
        """
    @typing.overload
    def set_color(self, arg0: int, arg1: str) -> None:
        """
        Set color of certain qubit.

        Set color of many qubits.
        """
    @typing.overload
    def set_color(self, arg0: typing.Dict[int, str]) -> None: ...
    @typing.overload
    def set_position(self, arg0: int, arg1: float, arg2: float) -> None:
        """
        Set position of a certain qubit.

        Set position of many qubits.
        """
    @typing.overload
    def set_position(self, arg0: typing.Dict[int, typing.Tuple[float, float]]) -> None: ...
    def size(self) -> int:
        """
        Get total qubit number.
        """
    pass

class LinearQubits(QubitsTopology):
    """
    Linear qubit topology.
    """

    def __init__(self, n_qubits: int) -> None:
        """
        Initialize a linear qubit topology.
        """
    pass

class QubitNode:
    """
    Qubit node.
    """

    def __gt__(self, other: QubitNode) -> QubitNode:
        """
        Disconnect with other qubit node and return rhs.
        """
    def __init__(self, id: int, color: str = '#000000', poi_x: float = 0.0, poi_y: float = 0.0) -> None:
        """
        Initialize a qubit node.
        """
    def __lshift__(self, other: QubitNode) -> QubitNode:
        """
        Connect with other qubit node and return lhs.
        """
    def __lt__(self, other: QubitNode) -> QubitNode:
        """
        Disconnect with other qubit node and return lhs.
        """
    def __rshift__(self, other: QubitNode) -> QubitNode:
        """
        Connect with other qubit node and return rhs.
        """
    def set_poi(self, poi_x: float, poi_y: float) -> None:
        """
        Set the position of this qubit.
        """
    @property
    def color(self) -> str:
        """
        Color of this qubit.

        :type: str
        """
    @property
    def id(self) -> int:
        """
        Index of this qubit.

        :type: int
        """
    @property
    def poi_x(self) -> float:
        """
        X position of this qubit.

        :type: float
        """
    @property
    def poi_y(self) -> float:
        """
        Y position of this qubit.

        :type: float
        """
    pass

class GridQubits(QubitsTopology):
    """
    Grid qubit topology.
    """

    def __init__(self, n_row: int, n_col: int) -> None:
        """
        Initialize a grid topology with row and col number.
        """
    def n_col(self) -> int:
        """
        Get column number.
        """
    def n_row(self) -> int:
        """
        Get row number.
        """
    pass
