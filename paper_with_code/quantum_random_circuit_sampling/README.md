Here’s an instructions for generating and utilizing tensor networks for random quantum circuits, as well as performing multi-GPU contractions:

Some files are in https://rec.ustc.edu.cn/share/b11ea090-653e-11ef-b54c-e12bd735bf53

(`measurements_n53_m20_s0_e0_pABCDCDAB.txt`, `640G_scheme_n53_m20.pt`, `640G_scheme_n53_m20_ABCDCDAB_3000000_einsum_10_open.pt`, `nsch640_split512_mg3_open10.pt`)

---

### Step 1: Generate Tensor Network and Contraction Order

1. **Define Random Quantum Circuit**:
    - Use `circuit_n53_m20_s0_e0_pABCDCDAB.py` to define the random quantum circuit. 
    - `QUBIT_ORDER` specifies the qubit order.
    - `CIRCUIT` specifies the quantum circuit structure.

2. **Alternate Circuit Definitions**:
    - `MindQuantum Circuit/circuit_n53_m20_s0_e0_pABCDCDAB_MindQuantum.py`: Defines the same circuit using the MindQuantum package.
    - `MindQuantum Circuit/qasm_circuit_n53_m20_s0_e0_pABCDCDAB.txt`: Defines the same circuit using QASM language.

3. **Transform Circuit to Tensor Network**:
    - Use `load_circuits.py` to define the `QuantumCircuit` class, which transforms the quantum circuit into a tensor network.

4. **Measurements Data**:
    - `measurements_n53_m20_s0_e0_pABCDCDAB.txt` contains 3 million bitstrings with calculated amplitudes.

5. **Generate Tensor Network and Contraction Order**:
    - Run `search_order.py` to generate the tensor network and contraction order.
    - The output is saved in `640G_scheme_n53_m20.pt`.

6. **Identify Open Qubit and Slicing Bond**:
    - Use `search_open_qubit.ipynb` to determine the ID of the open qubit and the slicing bond.
    - Modify the contraction order in `640G_scheme_n53_m20.pt` by removing the open qubit.

7. **Generate Contraction Task with Sample Space**:
    - Run `open_qubits.py` to create the contraction task with sample space (correlative space).
    - Save the new contraction task in `640G_scheme_n53_m20_ABCDCDAB_3000000_einsum_10_open.pt`.

### Step 2: Generate Multi-GPU Contraction Scheme

1. **Create Multi-GPU Contraction Scheme**:
    - Run `raw2nschrev-resingle.ipynb` to generate the multi-GPU contraction scheme.
    - Save the scheme in `nsch640_split512_mg3_open10.pt`.

### Step 3: Contract Using Multi-GPU

1. **Perform Multi-GPU Contraction**:
    - Run the script `mg_logtime-sg.py` using the following command:
      ```bash
      torchrun --nproc-per-node=8 mg_logtime-sg.py
      ```
    - The results will be saved in `sum_{ntask}.pt`.

---