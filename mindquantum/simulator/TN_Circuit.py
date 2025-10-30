import numpy as np

def _infer_nqubits_from_state(state) -> int:
    L = len(state)
    if L & (L - 1) != 0:
        raise ValueError("初始态长度不是 2 的整数次幂，无法对应量子比特数。")
    return int(np.log2(L))

def _bitstring_to_state(bitstr: str):
    n = len(bitstr)
    idx = int(bitstr, 2)
    state = np.zeros(2**n, dtype=np.complex128)
    state[idx] = 1.0
    return state

def _normalize_state(state: np.ndarray):
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("初始态范数为 0。")
    return state / norm

def mq_to_quimb(mq_circ,
                params: dict | None = None,
                init_state=None,
                init_state_fn=None,
                mq_sim=None):
    """
    将 MindQuantum Circuit 转成 quimb Circuit，并可选设定初始态。

    参数
    ----
    mq_circ : mindquantum.core.circuit.Circuit
    params : dict | None
        符号参数绑定（QASM2 不支持符号，需先数值化）。
    init_state : str | np.ndarray | None
        - str: 比特串，例如 "0101" -> |0101⟩
        - np.ndarray: 维度 2^n 的复杂态向量
        - None: 不设置，quimb 默认 |0...0⟩
    init_state_fn : callable | None
        回调：fn(n_qubits) -> np.ndarray，自定义生成初态。
    mq_sim : mindquantum.Simulator | None
        若提供，则会尝试从 mq_sim.get_qs() 读取当前态作为初始态。

    返回
    ----
    qc : quimb.tensor.Circuit
    """
    # --- 1) 参数绑定（与之前相同，做多版本兜底） ---
    if params:
        try:
            mq_circ = mq_circ.apply_value(params)
        except AttributeError:
            try:
                from mindquantum.core.parameterresolver import ParameterResolver
                pr = ParameterResolver(params)
                mq_circ = mq_circ.apply_value(pr)
            except Exception as e:
                raise RuntimeError("参数绑定失败，请检查 MindQuantum 版本与 API。") from e

    # --- 2) MQ -> QASM2 ---
    try:
        from mindquantum.io import OpenQASM
        qasm_str = OpenQASM().to_string(mq_circ)
    except Exception as e:
        raise RuntimeError("MindQuantum -> OpenQASM 导出失败，可能含有 QASM2 不支持的门。") from e

    # --- 3) quimb 读取 QASM ---
    from quimb.tensor import Circuit as QCircuit
    try:
        qc = QCircuit.from_openqasm2_str(qasm_str)
    except Exception as e:
        # 简单清理注释与空行再试
        cleaned = "\n".join(ln for ln in qasm_str.splitlines()
                            if not ln.strip().startswith("//")).strip()
        qc = QCircuit.from_openqasm2_str(cleaned)

    # --- 4) 设定初始态（可选） ---
    # 优先级：mq_sim -> init_state -> init_state_fn -> 默认
    psi0 = None

    if mq_sim is not None:
        # 从 MQ 模拟器读取（需要你在外部把 sim 调到想要的“初始时刻”）
        if hasattr(mq_sim, "get_qs"):
            psi0 = np.asarray(mq_sim.get_qs(), dtype=np.complex128).reshape(-1)
        else:
            raise RuntimeError("提供了 mq_sim，但未找到 get_qs() 方法。")

    if psi0 is None and init_state is not None:
        if isinstance(init_state, str):
            psi0 = _bitstring_to_state(init_state)
        else:
            psi0 = np.asarray(init_state, dtype=np.complex128).reshape(-1)

    if psi0 is None and init_state_fn is not None:
        # 从 quimb 电路对象估计比特数（其内部保存了 nqubits）
        try:
            n_qubits = qc.N
        except AttributeError:
            # 若 quimb 版本没有 N，可从已解析到的最大比特索引推断
            n_qubits = max(qc.qubits) + 1
        psi0 = np.asarray(init_state_fn(n_qubits), dtype=np.complex128).reshape(-1)

    # 若设置了 psi0，则规范化并检查维度匹配
    if psi0 is not None:
        psi0 = _normalize_state(psi0)
        # 校验维度是否匹配电路比特数
        n_from_state = _infer_nqubits_from_state(psi0)
        try:
            n_qubits = qc.N
        except AttributeError:
            n_qubits = max(qc.qubits) + 1
        if n_from_state != n_qubits:
            raise ValueError(
                f"初始态尺寸不匹配：state 有 {n_from_state} 比特，但电路是 {n_qubits} 比特。")
        # 挂到 quimb 电路
        qc.psi0 = psi0

    return qc


def TN_Circuit(mq_circ,
                params: dict | None = None,
                init_state=None,
                init_state_fn=None,
                mq_sim=None):
    return mq_to_quimb(mq_circ,
                params,
                init_state,
                init_state_fn,
                mq_sim)