#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mindquantum
from mindquantum.core.circuit import Circuit
from typing import List, Union, Dict, Callable, Any, Optional, Tuple, Iterable
from collections import OrderedDict, defaultdict
import copy
import itertools
import math

import numpy as np
import retworkx as rx

import numpy as np                             # 导入numpy库并简写为np
from mindquantum.simulator import Simulator    # 从mindquantum.simulator中导入Simulator类
from mindquantum.core.gates import BasicGate,X, H, RY,Z ,RX,RZ   # 导入量子门H, X, RY


# In[ ]:





# In[2]:


#定义DAGInNode和DAGOutNode
def _condition_as_indices(operation, bit_indices):
    cond = getattr(operation, "condition", None)
    if cond is None:
        return None
    bits, value = cond
    indices = [bit_indices[bits]] if isinstance(bits, list) else [bit_indices[x] for x in bits]
    return indices, value


class DAGNode:
    """Parent class for DAGOpNode, DAGInNode, and DAGOutNode."""

    __slots__ = ["_node_id"]

    def __init__(self, nid=-1):
        """Create a node"""
        self._node_id = nid

    def __lt__(self, other):
        return self._node_id < other._node_id

    def __gt__(self, other):
        return self._node_id > other._node_id

    def __str__(self):
        # TODO is this used anywhere other than in DAG drawing?
        # needs to be unique as it is what pydot uses to distinguish nodes
        return str(id(self))

    @staticmethod
    def semantic_eq(node1, node2, bit_indices1=None, bit_indices2=None):
        
        #Check if DAG nodes are considered equivalent.
        
        if bit_indices1 is None or bit_indices2 is None:
            warnings.warn(
                "DAGNode.semantic_eq now expects two bit-to-circuit index "
                "mappings as arguments. To ease the transition, these will be "
                "pre-populated based on the values found in Bit.index and "
                "Bit.register. However, this behavior is deprecated and a future "
                "release will require the mappings to be provided as arguments.",
                DeprecationWarning,
            )

            bit_indices1 = {arg: arg for arg in node1.qargs + node1.cargs}
            bit_indices2 = {arg: arg for arg in node2.qargs + node2.cargs}

        if isinstance(node1, DAGOpNode) and isinstance(node2, DAGOpNode):
            node1_qargs = [bit_indices1[qarg] for qarg in node1.qargs]
            node1_cargs = [bit_indices1[carg] for carg in node1.cargs]

            node2_qargs = [bit_indices2[qarg] for qarg in node2.qargs]
            node2_cargs = [bit_indices2[carg] for carg in node2.cargs]

            # For barriers, qarg order is not significant so compare as sets
            if node1.op.name == node2.op.name and node1.name in {"barrier", "swap"}:
                return set(node1_qargs) == set(node2_qargs)

            return (
                node1_qargs == node2_qargs
                and node1_cargs == node2_cargs
                and (
                    _condition_as_indices(node1.op, bit_indices1)
                    == _condition_as_indices(node2.op, bit_indices2)
                )
                and node1.op == node2.op
            )
        if (isinstance(node1, DAGInNode) and isinstance(node2, DAGInNode)) or (
            isinstance(node1, DAGOutNode) and isinstance(node2, DAGOutNode)
        ):
            return bit_indices1.get(node1.wire, None) == bit_indices2.get(node2.wire, None)

        return False


class DAGOpNode(DAGNode):
    """Object to represent an Instruction at a node in the DAGCircuit."""

    __slots__ = ["op", "qargs", "cargs", "sort_key"]

    def __init__(self, op, qargs=(), cargs=()):
        """Create an Instruction node"""
        super().__init__()
        self.op = op
        self.qargs = tuple(qargs)
        self.cargs = tuple(cargs)
        self.sort_key = str(self.qargs)

    @property
    def name(self):
        """Returns the Instruction name corresponding to the op for this node"""
        return self.op.name

    @name.setter
    def name(self, new_name):
        """Sets the Instruction name corresponding to the op for this node"""
        self.op.name = new_name

    def __repr__(self):
        """Returns a representation of the DAGOpNode"""
        return f"DAGOpNode(op={self.op}, qargs={self.qargs}, cargs={self.cargs})"


class DAGInNode(DAGNode):
    """Object to represent an incoming wire node in the DAGCircuit."""

    __slots__ = ["wire", "sort_key"]

    def __init__(self, wire):
        """Create an incoming node"""
        super().__init__()
        self.wire = wire
        # TODO sort_key which is used in dagcircuit.topological_nodes
        # only works as str([]) for DAGInNodes. Need to figure out why.
        self.sort_key = str([])

    def __repr__(self):
        """Returns a representation of the DAGInNode"""
        return f"DAGInNode(wire={self.wire})"


class DAGOutNode(DAGNode):
    """Object to represent an outgoing wire node in the DAGCircuit."""

    __slots__ = ["wire", "sort_key"]

    def __init__(self, wire):
        """Create an outgoing node"""
        super().__init__()
        self.wire = wire
        # TODO sort_key which is used in dagcircuit.topological_nodes
        # only works as str([]) for DAGOutNodes. Need to figure out why.
        self.sort_key = str([])

    def __repr__(self):
        """Returns a representation of the DAGOutNode"""
        return f"DAGOutNode(wire={self.wire})"


# In[5]:


class DAGCircuit:
    def __init__(self):
        # Circuit name.  Generally, this corresponds to the name
        # of the QuantumCircuit from which the DAG was generated.
        self.name = None

        # Circuit metadata
        self.metadata = None

        # Set of wires (Register,idx) in the dag
        self._wires = set()

        # Map from wire (Register,idx) to input nodes of the graph
        self.input_map = OrderedDict()

        # Map from wire (Register,idx) to output nodes of the graph
        self.output_map = OrderedDict()

        # Directed multigraph whose nodes are inputs, outputs, or operations.
        self._multi_graph = rx.PyDAG()

        # Map of qreg/creg name to Register object.
        self.qregs = OrderedDict()
        self.cregs = OrderedDict()

        # List of Qubit/Clbit wires that the DAG acts on.
        self.qubits = []
        self.clbits = []

        self._global_phase = 0
        self._calibrations = defaultdict(dict)

        self._op_names = {}

        self.duration = None
        self.unit = "dt"
        
    def add_qubits(self, qubits):
        """Add individual qubit wires."""
        if not isinstance(qubits, list):
            raise TypeError(f"qubits need a list, but get {type(qubits)}!")

        duplicate_qubits = set(self.qubits).intersection(qubits)
        if duplicate_qubits:
            raise TypeError("qubits %s already exist！" % duplicate_qubits)

        self.qubits.extend(qubits)
        for qubit in qubits:
            self._add_wire(qubit)
            
    def add_clbits(self, clbits):
        """Add individual clbit wires."""
        if not isinstance(clbits, list):
            raise TypeError(f"clbits need a list, but get {type(clbits)}!")

        duplicate_clbits = set(self.clbits).intersection(clbits)
        if duplicate_clbits:
            raise TypeError("clbits %s already exist！" % duplicate_clbits)

        self.clbits.extend(clbits)
        for clbit in clbits:
            self._add_wire(clbit)
            
    def _add_wire(self, wire):
        """Add a qubit or bit to the circuit.
        Args:
            wire (Bit): the wire to be added
            This adds a pair of in and out nodes connected by an edge.
        Raises:
            DAGCircuitError: if trying to add duplicate wire
        """
        if wire not in self._wires:
            self._wires.add(wire)

            inp_node = DAGInNode(wire=wire)
            outp_node = DAGOutNode(wire=wire)
            input_map_id, output_map_id = self._multi_graph.add_nodes_from([inp_node, outp_node])
            inp_node._node_id = input_map_id
            outp_node._node_id = output_map_id
            self.input_map[wire] = inp_node
            self.output_map[wire] = outp_node
            self._multi_graph.add_edge(inp_node._node_id, outp_node._node_id, wire)
        else:
            raise TypeError(f"duplicate wire {wire}")
            
            
    def _add_op_node(self, op, qargs, cargs):
        """Add a new operation node to the graph and assign properties.
        Args:
            op (qiskit.circuit.Instruction): the operation associated with the DAG node
            qargs (list[Qubit]): list of quantum wires to attach to.
            cargs (list[Clbit]): list of classical wires to attach to.
        Returns:
            int: The integer node index for the new op node on the DAG
        """
        # Add a new operation node to the graph
        new_node = DAGOpNode(op=op, qargs=qargs, cargs=cargs)
        node_index = self._multi_graph.add_node(new_node)
        new_node._node_id = node_index
        self._increment_op(op)
        return node_index
    
    
    def apply_operation_back(self, op, qargs=(), cargs=()):
        """Apply an operation to the output of the circuit.
        Args:
            op (qiskit.circuit.Instruction): the operation associated with the DAG node
            qargs (tuple[Qubit]): qubits that op will be applied to
            cargs (tuple[Clbit]): cbits that op will be applied to
        Returns:
            DAGOpNode: the node for the op that was added to the dag
        Raises:
            DAGCircuitError: if a leaf node is connected to multiple outputs
        """
        qargs = tuple(qargs) if qargs is not None else ()
        cargs = tuple(cargs) if cargs is not None else ()

        node_index = self._add_op_node(op, qargs, cargs)

        # Add new in-edges from predecessors of the output nodes to the
        # operation node while deleting the old in-edges of the output nodes
        # and adding new edges from the operation node to each output node

        al = [qargs]
        self._multi_graph.insert_node_on_in_edges_multiple(
            node_index, [self.output_map[q]._node_id for q in itertools.chain(*al)]
        )
        return self._multi_graph[node_index]
    
            
    def _increment_op(self, op):
        if op.name in self._op_names:
            self._op_names[op.name] += 1
        else:
            self._op_names[op.name] = 1

    def _decrement_op(self, op):
        if self._op_names[op.name] == 1:
            del self._op_names[op.name]
        else:
            self._op_names[op.name] -= 1

    def _add_op_node(self, op, qargs, cargs):
        """Add a new operation node to the graph and assign properties.
        Args:
            op (qiskit.circuit.Instruction): the operation associated with the DAG node
            qargs (list[Qubit]): list of quantum wires to attach to.
            cargs (list[Clbit]): list of classical wires to attach to.
        Returns:
            int: The integer node index for the new op node on the DAG
        """
        # Add a new operation node to the graph
        new_node = DAGOpNode(op=op, qargs=qargs, cargs=cargs)
        node_index = self._multi_graph.add_node(new_node)
        new_node._node_id = node_index
        self._increment_op(op)
        return node_index
    
    def remove_op_node(self, node):
        """Remove an operation node n.
        Add edges from predecessors to successors.
        """

        self._multi_graph.remove_node_retain_edges(
            node._node_id, use_outgoing=False, condition=lambda edge1, edge2: edge1 == edge2
        )
        self._decrement_op(node.op)
    
    def get_all_opnodes(self):
        def _key(x):
            return x.sort_key
        all_nodes=iter(rx.lexicographical_topological_sort(self._multi_graph, key=_key))
        return (nd for nd in all_nodes if isinstance(nd, DAGOpNode))
    
    def get_all_nodes(self):
        def _key(x):
            return x.sort_key
        all_nodes=iter(rx.lexicographical_topological_sort(self._multi_graph, key=_key))
        return all_nodes
    
    def compose(self, other, qubits=None, clbits=None, front=False, inplace=True):
        #Compose the ``other`` circuit onto the output of this circuit.
        # number of qubits and clbits must match number in circuit or None
        identity_qubit_map = dict(zip(other.qubits, self.qubits))
        identity_clbit_map = dict(zip(other.clbits, self.clbits))
        if qubits is None:
            qubit_map = identity_qubit_map
        else:
            qubit_map = {
                other.qubits[i]: (self.qubits[q] if isinstance(q, int) else q)
                for i, q in enumerate(qubits)
            }
        if clbits is None:
            clbit_map = identity_clbit_map
        else:
            clbit_map = {
                other.clbits[i]: (self.clbits[c] if isinstance(c, int) else c)
                for i, c in enumerate(clbits)
            }
        edge_map = {**qubit_map, **clbit_map} or None

        # if no edge_map, try to do a 1-1 mapping in order
        if edge_map is None:
            edge_map = {**identity_qubit_map, **identity_clbit_map}

        # Check the edge_map for duplicate values
        if len(set(edge_map.values())) != len(edge_map):
            raise DAGCircuitError("duplicates in wire_map")

        # Compose
        if inplace:
            dag = self
        else:
            dag = copy.deepcopy(self)

        for nd in other.topological_nodes():
            if isinstance(nd, DAGInNode):
                # if in edge_map, get new name, else use existing name
                m_wire = edge_map.get(nd.wire, nd.wire)
                
            elif isinstance(nd, DAGOutNode):
                # ignore output nodes
                pass
            elif isinstance(nd, DAGOpNode):
                m_qargs = list(map(lambda x: edge_map.get(x, x), nd.qargs))
                m_cargs = list(map(lambda x: edge_map.get(x, x), nd.cargs))
                op = nd.op.copy()

        if not inplace:
            return dag
        else:
            return None
        
    def copy_empty_like(self):
        """Return a copy of self with the same structure but empty.
        """
        target_dag = DAGCircuit()
        target_dag.name = self.name

        target_dag.add_qubits(self.qubits)
        target_dag.add_clbits(self.clbits)

        return 
    
        
    def serial_layers(self):
        """Yield a layer for all gates of this circuit.
        A serial layer is a circuit with one gate. The layers have the
        same structure as in layers().
        """
        for next_node in self.get_all_opnodes():
            new_layer = self.copy_empty_like()

            # Save the support of the operation we add to the layer
            support_list = []
            # Operation data
            op = copy.copy(next_node.op)
            qargs = copy.copy(next_node.qargs)

            # Add node to new_layer
            new_layer.apply_operation_back(op, qargs, cargs)
            yield new_layer
    
    def two_qubit_ops(self):
        """Get list of 2 qubit operations. Ignore directives like snapshot and barrier."""
        ops = []
        for node in self.get_all_opnodes(include_directives=False):
            if len(node.qargs) == 2:
                ops.append(node)
        return ops
    
    
    def collect_1q_runs(self):
        """Return a set of non-conditional runs of 1q "op" nodes."""

        def filter_fn(node):
            return (
                isinstance(node, DAGOpNode)
                and len(node.qargs) == 1
                and len(node.cargs) == 0
                and isinstance(node.op, BasicGate)
            )

        return rx.collect_runs(self._multi_graph, filter_fn)
     
from retworkx.visualization import graphviz_draw
def dag_drawer(dag, scale=0.7, filename=None, style="color"):
    # NOTE: use type str checking to avoid potential cyclical import
    # the two tradeoffs ere that it will not handle subclasses and it is
    # slower (which doesn't matter for a visualization function)
    type_str = str(type(dag))
    if "DAGDependency" in type_str:
        graph_attrs = {"dpi": str(100 * scale)}

        def node_attr_func(node):
            if style == "plain":
                return {}
            if style == "color":
                n = {}
                n["label"] = str(node.node_id) + ": " + str(node.name)
                if node.name == "measure":
                    n["color"] = "blue"
                    n["style"] = "filled"
                    n["fillcolor"] = "lightblue"
                if node.name == "barrier":
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "green"
                if getattr(node.op, "_directive", False):
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "red"
                if getattr(node.op, "condition", None):
                    n["label"] = str(node.node_id) + ": " + str(node.name) + " (conditional)"
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "lightgreen"
                return n
            else:
                raise VisualizationError("Unrecognized style %s for the dag_drawer." % style)

        edge_attr_func = None

    else:
        qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        clbit_indices = {bit: index for index, bit in enumerate(dag.clbits)}
        qu_indices={index: bit for index, bit in enumerate(dag.qubits)}
        register_bit_labels = {
            bit: f"[{idx}]"
            for (idx, bit) in enumerate(dag.qubits)
        }

        graph_attrs = {"dpi": str(100 * scale)}

        def node_attr_func(node):
            if style == "plain":
                return {}
            if style == "color":
                n = {}
                if isinstance(node, DAGOpNode):
                    n["label"] = node.name
                    n["color"] = "blue"
                    n["style"] = "filled"
                    n["fillcolor"] = "yellow"
                if isinstance(node, DAGInNode):
                    n["label"] = "q"+str(node.wire)
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "cyan"
                if isinstance(node, DAGOutNode):
                    n["label"] = "q"+str(node.wire)
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "red"
                return n
            else:
                print("VisualizationError:Invalid style")
                #raise VisualizationError("Invalid style %s" % style)

        def edge_attr_func(edge):
            e = {}
            label="label"
            e["label"] = f"q{qu_indices[edge]}"
            return e

    image_type = None
    if filename:
        if "." not in filename:
            print("InvalidFileError")
            #raise InvalidFileError("Parameter 'filename' must be in format 'name.extension'")
        image_type = filename.split(".")[-1]
    return graphviz_draw(
        dag._multi_graph,
        node_attr_func,
        edge_attr_func,
        graph_attrs,
        filename,
        image_type,
    )


# In[13]:


def circuit_to_dag(circuit):
    """Build a ``DAGCircuit`` object from a ``Circuit``.
    Args:
        circuit (Circuit): the input circuit.
    Return:
        DAGCircuit: the DAG representing the input circuit.
    """
    dagcircuit = DAGCircuit()

    dagcircuit.add_qubits(circuit.all_qubits.keys())

    for instruction in circuit:
        newop=dagcircuit.apply_operation_back(
            copy.deepcopy(instruction), instruction.obj_qubits+instruction.ctrl_qubits,[]
        )
        #print(newop._node_id)
    return dagcircuit

def circuit_to_dag1(circuit):
    """Build a ``DAGCircuit`` object from a ``Circuit``.
    Args:
        circuit (Circuit): the input circuit.
    Return:
        DAGCircuit: the DAG representing the input circuit.
    """
    dagcircuit = DAGCircuit()

    dagcircuit.add_qubits(circuit.all_qubits.keys())
    li=[]
    for instruction in circuit:
        newop=dagcircuit.apply_operation_back(
            copy.deepcopy(instruction), instruction.obj_qubits+instruction.ctrl_qubits,[]
        )
        li.append(newop._node_id)
    return dagcircuit,li


# In[9]:


def dag_to_circuit(dag):
    circuit=Circuit()
    for node in dag.get_all_opnodes():
        circuit.append(copy.deepcopy(node.op))

    return circuit


# In[10]:





