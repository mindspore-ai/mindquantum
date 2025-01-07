import copy
import torch as tc
from torch import nn

mps_propertie_keys = ['oee', 'eosp_ordering', 'eosp_oee_av', 'qs_number']


class MPS_basic:

    def __init__(self, tensors=None, para=None, properties=None):
        self.name = 'MPS'
        self.para = dict()
        self.input_paras(para)
        self.center = -1  # 正交中心（-1代表不具有正交中心）

        # 以下属性参考self.update_properties（通过properties输入）
        self.oee = None  # 单点纠缠熵（所有点）
        self.eosp_ordering = None  # EOSP采样顺序（所有点）
        self.eosp_oee_av = None  # EOSP采样顺序每次的平均OEE
        self.qs_number = None  # Q稀疏性

        self.eps = self.para['eps']
        self.device = choose_device(self.para['device'])
        self.dtype = self.para['dtype']
        self.dc = self.para['dc']
        self.spin = self.para['spin']
        if tensors is None:
            if self.spin:
                self.tensors = random_mps_for_spin(self.para['length'], self.para['d'], self.para['chi'],
                                          self.para['boundary'], self.device, self.dtype)
            else:
                self.tensors = random_mps(self.para['length'], self.para['d'], self.para['chi'],
                                      self.para['boundary'], self.device, self.dtype)
        else:
            self.tensors = tensors
            self.length = len(self.tensors)
            self.to()

        self.orthogonalized_tensors = [x.clone() for x in self.tensors]
        self.update_attributes_para()
        self.update_properties(properties)

    def input_paras(self, para=None):
        para0 = {
            'length': 4,
            'd': 2,
            'chi': 3,
            'boundary': 'open',
            'eps': 0,
            'device': None,
            'dtype': tc.float64
        }
        if para is None:
            self.para = para0
        else:
            self.para = dict(para0, **para)

    def act_single_gate(self, gate, pos, unitary_gate=False):
        gate = gate.to(device=self.device)
        self.tensors[pos] = tc.einsum('ps,asb->apb', gate, self.tensors[pos].to(dtype=gate.dtype))
        if not unitary_gate:
            self.center = -1

    def act_single_gate_on_H_psi(self, gate, pos, unitary_gate=False):
        gate = gate.to(device=self.device)
        tensor = self.tensors[pos].clone()
        self.H_psi_tensors[pos] = tc.einsum('ps,asb->apb', gate, tensor.to(dtype=gate.dtype))
        if not unitary_gate:
            self.center = -1

    def act_ham_term(self, ham_term_tuple, unitary_gate=True):  # ham_term_tuple : (1.0, [('Z', 0), ('Z', 1)])
        for gate, pos in ham_term_tuple[1]:
            gate = pauli_operators(gate)
            self.act_single_gate(gate, pos, unitary_gate)
        if not unitary_gate:
            self.center = -1

    def bipartite_entanglement(self, nt, normalize=False):
        # 从第nt个张量右边断开，计算纠缠
        # 计算过程中，会对MPS进行规范变换
        if self.center <= nt:
            self.center_orthogonalization(nt, 'qr', dc=-1, normalize=normalize)
            lm = tc.linalg.svdvals(self.tensors[nt].reshape(
                -1, self.tensors[nt].shape[-1]))
        else:
            self.center_orthogonalization(nt + 1, 'qr', dc=-1, normalize=normalize)
            lm = tc.linalg.svdvals(self.tensors[nt+1].reshape(
                self.tensors[nt+1].shape[0], -1))
        return lm

    def copy_properties(self, mps, properties=None):
        if properties is None:
            properties = mps_propertie_keys
        for x in properties:
            setattr(self, x, getattr(mps, x))

    def correct_device(self):
        self.device = choose_device(self.device)
        self.to()

    def clone_mps(self):
        tensors = [x.clone() for x in self.tensors]
        mps1 = MPS_basic(tensors=tensors, para=copy.deepcopy(self.para))
        for x in mps_propertie_keys:
            setattr(mps1, x, getattr(self, x))
        return mps1


    def clone_tensors(self):
        self.tensors = [x.clone() for x in self.tensors]

    def center_orthogonalization(self, c, way='svd', dc=-1, normalize=False, is_tensors_leaf=True):
        if c == -1:
            c = len(self.tensors) - 1
        if self.center < -0.5:
            self.orthogonalize_n1_n2(0, c, way, dc, normalize, is_tensors_leaf=is_tensors_leaf)
            self.orthogonalize_n1_n2(len(self.tensors)-1, c, way, dc, normalize, is_tensors_leaf=is_tensors_leaf)
        elif self.center != c:
            self.orthogonalize_n1_n2(self.center, c, way, dc, normalize, is_tensors_leaf=is_tensors_leaf)
        self.center = c
        if normalize:
            self.normalize_central_tensor(is_tensors_leaf=is_tensors_leaf)
            if not is_tensors_leaf:
                s = self.orthogonalized_tensors[0].shape
                self.orthogonalized_tensors[0] = tc.tensordot(self.orthogonalized_tensors[0], tc.eye(s[0], s[2], device=self.device, dtype=self.dtype), [[0], [0]])
                self.orthogonalized_tensors[0] = self.orthogonalized_tensors[0] / self.orthogonalized_tensors[0].norm()

                s = self.orthogonalized_tensors[-1].shape
                self.orthogonalized_tensors[-1] = tc.tensordot(self.orthogonalized_tensors[-1], tc.eye(s[0], s[2], device=self.device, dtype=self.dtype), [[2], [1]])
                self.orthogonalized_tensors[-1] = self.orthogonalized_tensors[-1] / self.orthogonalized_tensors[-1].norm()


    def check_center_orthogonality(self, prt=True):
        if self.center < -0.5:
            if prt:
                print('MPS NOT in center-orthogonal form!')
        else:
            err = check_center_orthogonality(self.tensors, self.center, prt=prt)
            return err

    def check_virtual_dims(self):
        for n in range(len(self.tensors)-1):
            assert self.tensors[n].shape[-1] == self.tensors[n+1].shape[0]
        assert self.tensors[0].shape[0] == self.tensors[-1].shape[-1]


    def find_max_virtual_dim(self):
        dims = [x.shape[0] for x in self.tensors]
        return max(dims)

    def full_tensor(self):
        return full_tensor(self.tensors)

    def move_center_one_step(self, direction, decomp_way, dc, normalize, is_tensors_leaf):
        if direction.lower() in ['right', 'r']:
            if -0.5 < self.center < self.length-1:
                self.orthogonalize_left2right(self.center, decomp_way, dc, normalize, is_tensors_leaf=is_tensors_leaf)
                self.center += 1
            else:
                print('Error: cannot move center left as center = ' + str(self.center))
        elif direction.lower() in ['left', 'l']:
            if self.center > 0:
                self.orthogonalize_right2left(self.center, decomp_way, dc, normalize, is_tensors_leaf=is_tensors_leaf)
                self.center -= 1
            else:
                print('Error: cannot move center right as center = ' + str(self.center))

    def normalize_central_tensor(self, normalize=True, is_tensors_leaf=True):
        if is_tensors_leaf:
            norm = self.orthogonalized_tensors[self.center].norm().detach()
        else:
            norm = self.tensors[self.center].norm().detach()
        if normalize:
            if is_tensors_leaf:
                self.orthogonalized_tensors[self.center] = self.orthogonalized_tensors[self.center].clone() / norm
            else:
                self.tensors[self.center] = self.tensors[self.center].clone() / norm
        return norm



    def orthogonalize_left2right(self, nt, way, dc=-1,
                                 normalize=False, is_tensors_leaf=True):
        # dc=-1意味着不进行裁剪
        assert nt < len(self.tensors)-1
        s = self.orthogonalized_tensors[nt].shape
        if 0 < dc < s[-1]:
            # In this case, truncation is required
            way = 'svd'
            if_trun = True
        else:
            if_trun = False

        tensor = self.tensors[nt].reshape(-1, s[-1]).to('cpu')
        if way.lower() == 'svd':
            u, lm, v = tc.linalg.svd(tensor,
                                     full_matrices=False)
            lm = lm.to(dtype=u.dtype)
            if if_trun:
                u = u[:, :dc].to(device=self.device)
                r = tc.diag(lm[:dc]).to(device=self.device).mm(
                    v[:dc, :].to(device=self.device))
            else:
                u = u.to(device=self.device)
                r = tc.diag(lm).to(device=self.device).mm(v.to(device=self.device))
        else:
            u, r = tc.linalg.qr(tensor)
            lm = None
            u, r = u.to(device=self.device), r.to(device=self.device)
        if is_tensors_leaf:
            self.orthogonalized_tensors[nt] = u.reshape(s[0], s[1], -1)
        else:
            self.tensors[nt] = u.reshape(s[0], s[1], -1)
        if normalize:
            r = r / tc.norm(r)
        if is_tensors_leaf:
            self.orthogonalized_tensors[nt + 1] = tc.tensordot(
                r.to(dtype=self.orthogonalized_tensors[nt + 1].dtype), self.orthogonalized_tensors[nt + 1], [[1], [0]])
        else:
            self.tensors[nt+1] = tc.tensordot(
            r.to(dtype=self.tensors[nt+1].dtype), self.tensors[nt+1], [[1], [0]])
        return lm

    def orthogonalize_right2left(self, nt, way, dc=-1, normalize=False, is_tensors_leaf=True):
        # dc=-1意味着不进行裁剪
        assert nt > 0
        s = self.orthogonalized_tensors[nt].shape
        if 0 < dc < s[0]:
            # In this case, truncation is required
            way = 'svd'
            if_trun = True
        else:
            if_trun = False

        tensor = self.orthogonalized_tensors[nt].reshape(s[0], -1).t().to('cpu')
        if way.lower() == 'svd':
            u, lm, v = tc.linalg.svd(tensor, full_matrices=False)
            lm = lm.to(dtype=u.dtype)
            if if_trun:
                u = u[:, :dc].to(device=self.device)
                r = tc.diag(lm[:dc]).to(device=self.device).mm(v[:dc, :].to(self.device))
            else:
                u = u.to(device=self.device)
                r = tc.diag(lm).to(device=self.device).mm(v.to(device=self.device))
        else:
            u, r = tc.linalg.qr(tensor)
            lm = None
            u, r = u.to(device=self.device), r.to(device=self.device)
        if is_tensors_leaf:
            self.orthogonalized_tensors[nt] = u.t().reshape(-1, s[1], s[2])
        else:
            self.tensors[nt] = u.t().reshape(-1, s[1], s[2])
        if normalize:
            r = r / tc.norm(r)

        if is_tensors_leaf:
            self.orthogonalized_tensors[nt - 1] = tc.tensordot(self.orthogonalized_tensors[nt - 1], r.to(dtype=self.orthogonalized_tensors[nt - 1].dtype),
                                                [[2], [1]])
        else:
            self.tensors[nt - 1] = tc.tensordot(self.tensors[nt - 1], r.to(dtype=self.tensors[nt-1].dtype), [[2], [1]])
        return lm

    def orthogonalize_n1_n2(self, n1, n2, way, dc, normalize, is_tensors_leaf):
        if n1 < n2:
            for nt in range(n1, n2, 1):
                self.orthogonalize_left2right(nt, way, dc, normalize, is_tensors_leaf=is_tensors_leaf)
        elif n1 > n2:
            for nt in range(n1, n2, -1):
                self.orthogonalize_right2left(nt, way, dc, normalize, is_tensors_leaf=is_tensors_leaf)

    def project_qubit_nt(self, nt, state):
        states_vecs = (type(state) is tc.Tensor) and (state.numel() == 2)
        if states_vecs:
            self.tensors[nt] = tc.tensordot(self.tensors[nt], state, [[1], [0]])
        else:
            self.tensors[nt] = self.tensors[nt][:, state, :]
        if len(self.tensors) > 1:
            if nt == 0:  # contract to 1st tensor
                self.tensors[1] = tc.tensordot(self.tensors[0], self.tensors[1], [[1], [0]])
            else:  # contract to the left tensor
                self.tensors[nt-1] = tc.tensordot(self.tensors[nt-1], self.tensors[nt], [[-1], [0]])
            self.tensors.pop(nt)

    def project_multi_qubits(self, pos, states):
        # Not central-orthogonalized; not normalized
        assert type(pos) is list
        states_vecs = (type(states) is tc.Tensor) and (states.ndimension() == 2)
        for n, p in enumerate(pos):
            if states_vecs:
                self.tensors[p] = tc.tensordot(self.tensors[p], states[:, n], [[1], [0]])
            else:
                self.tensors[p] = self.tensors[p][:, states[n], :]
        pos = sorted(copy.deepcopy(pos))
        for p in pos[len(pos):0:-1]:
            assert p > 0
            self.tensors[p - 1] = tc.tensordot(self.tensors[p - 1], self.tensors[p], [[-1], [0]])
            self.tensors.pop(p)
        if len(self.tensors) > 1:
            if pos[0] == 0:
                self.tensors[1] = tc.tensordot(self.tensors[0], self.tensors[1], [[-1], [0]])
            else:
                self.tensors[pos[0]-1] = tc.tensordot(
                    self.tensors[pos[0]-1], self.tensors[pos[0]], [[-1], [0]])
            self.tensors.pop(pos[0])
        self.center = -1

    def properties(self, prop=None, which_prop=None):
        if prop is None:
            prop = dict()
        if which_prop is None:
            which_prop = mps_propertie_keys
        elif type(which_prop) is str:
            which_prop = [which_prop]
        for x in which_prop:
            if getattr(self, x) is not None:
                prop[x] = getattr(self, x)
        return prop

    def tensors2ParameterList(self, eps=1.0):
        tensors = nn.ParameterList()
        for x in self.tensors:
            tensors.append(nn.Parameter(x * eps, requires_grad=True))
        self.tensors = tensors

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = choose_device(device)
        if dtype is not None:
            self.dtype = dtype
        tensors = [x.to(device=self.device, dtype=self.dtype) for x in self.tensors]
        self.tensors = tensors

    def two_body_RDM(self, pos):
        # 注：该函数会改变MPS的中心位置
        pos = sorted(pos)
        self.center_orthogonalization(c=pos[0], way='qr')
        self.normalize_central_tensor()
        v = tc.einsum('apb,aqd->pqbd',
                      self.tensors[pos[0]].conj(), self.tensors[pos[0]])
        for n in range(pos[0]+1, pos[1], 1):
            v = tc.einsum('pqac,asb,csd->pqbd',
                          v, self.tensors[n].conj(), self.tensors[n])
        rho = tc.einsum('pqac,asb,ckb->psqk',
                        v, self.tensors[pos[1]].conj(), self.tensors[pos[1]])
        s = rho.shape
        rho = rho.reshape(s[0]*s[1], -1)
        return rho

    def update_attributes_para(self):
        self.length = len(self.tensors)
        self.para['device'] = self.device
        self.para['dtype'] = self.dtype
        self.para['length'] = self.length

    def update_properties(self, properties):
        if type(properties) is dict:
            for x in mps_propertie_keys:
                if x in properties:
                    setattr(self, x, properties[x])





def check_center_orthogonality(tensors, center, prt=False):
    err = [None] * len(tensors)
    for n in range(center):
        s = tensors[n].shape
        tmp = tensors[n].reshape(-1, s[-1])
        tmp = tmp.t().conj().mm(tmp)
        err[n] = (tmp - tc.eye(tmp.shape[0], device=tensors[n].device,
                               dtype=tensors[n].dtype)).norm(p=1).item()
    for n in range(len(tensors)-1, center, -1):
        s = tensors[n].shape
        tmp = tensors[n].reshape(s[0], -1)
        tmp = tmp.mm(tmp.t().conj())
        err[n] = (tmp - tc.eye(tmp.shape[0], device=tensors[n].device,
                               dtype=tensors[n].dtype)).norm(p=1).item()

    if prt:
        print('Orthogonality check:')
        print('=' * 35)
        err_av = 0.0
        for n in range(len(tensors)):
            if err[n] is None:
                print('Site ' + str(n) + ':  center')
            else:
                print('Site ' + str(n) + ': ', err[n])
                err_av += err[n]
        print('-' * 35)
        print('Average error = %g' % (err_av / (len(tensors) - 1)))
        print('=' * 35)
    return err




def full_tensor(tensors):
    # 注：要求每个张量第0个指标为左虚拟指标，最后一个指标为右虚拟指标
    psi = tensors[0]
    for n in range(1, len(tensors)):
        psi = tc.tensordot(psi, tensors[n].to(dtype=psi.dtype), [[-1], [0]])
    if psi.shape[0] > 1:  # 周期边界
        psi = psi.permute([0, psi.ndimension()-1] + list(range(1, psi.ndimension()-1)))
        s = psi.shape
        psi = tc.einsum('aab->b', psi.reshape(s[0], s[1], -1))
        psi = psi.reshape(s[2:])
    else:
        psi = psi.squeeze()
    return psi

def random_mps(length, d, chi, boundary='open', device=None, dtype=tc.float64):
    device = choose_device(device)
    if boundary == 'open':
        tensors = [tc.randn((chi, d, chi), device=device, dtype=dtype, requires_grad=True)
                   for _ in range(length - 2)]
        return [tc.randn((1, d, chi), device=device, dtype=dtype, requires_grad=True)] + tensors + [
            tc.randn((chi, d, 1), device=device, dtype=dtype, requires_grad=True)]
    else:  # 周期边界MPS
        return [tc.randn((chi, d, chi), device=device, dtype=dtype, requires_grad=True)
                for _ in range(length)]


def random_mps_for_spin(length, d, chi, boundary='open', device=None, dtype=tc.float64):
    assert d == 2, "This function assumes d=2."

    def generate_hermitian_matrix(chi, device, dtype):
        """
        Generates a Hermitian matrix A where A * A^† = I
        """
        # Create a random matrix
        A = tc.randn((chi, chi), device=device, dtype=dtype, requires_grad=True)

        # If using complex numbers, generate the imaginary part
        if dtype.is_complex:
            A += 1j * tc.randn((chi, chi), device=device, dtype=dtype, requires_grad=True)

        # Create Hermitian matrix (A + A^†) / 2
        A_herm = (A + A.conj().T) / 2

        # Use QR decomposition to ensure orthogonality
        Q, R = tc.linalg.qr(A_herm)

        return Q

    def generate_tensor(chi, device, dtype):
        """
        Generates a tensor with Hermitian matrices for d=2
        """
        A = generate_hermitian_matrix(chi, device, dtype)
        B = generate_hermitian_matrix(chi, device, dtype)

        # Stack them into a tensor of shape (chi, d, chi)
        tensor = tc.stack([A, B], dim=1)
        return tensor

    if boundary == 'open':
        tensors = [generate_tensor(chi, device, dtype) for _ in range(length - 2)]
        return [generate_tensor(1, device, dtype)] + tensors + [generate_tensor(1, device, dtype)]
    else:  # 周期边界MPS
        return [generate_tensor(chi, device, dtype) for _ in range(length)]

def parse_hamiltonian(hamiltonian_str):
    """
    :param hamiltonian_str: eg. '1.0 [Z0 Z1] + 2.0 [X0 X1] + 3.0 [Y0 Y1]'
    :return: eg. [(1.0, [('Z', 0), ('Z', 1)]), (2.0, [('X', 0), ('X', 1)]), (3.0, [('Y', 0), ('Y', 1)])]
    """
    hamiltonian = []
    lines = hamiltonian_str.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 分离系数和操作符
        if ' ' in line:
            coefficient_str, operator_str = line.split(' ', 1)
        else:
            coefficient_str = line
            operator_str = ''

        # 提取系数
        coefficient_str = coefficient_str.strip().replace('[', '').replace(']', '')
        try:
            coefficient = float(coefficient_str)
        except ValueError:
            continue  # 跳过无效的系数

        # 提取操作符
        operator_str = operator_str.strip().replace('[', '').replace(']', '')
        operators = []
        if operator_str:
            for op in operator_str.split():
                if op and op != '+':
                    # 提取操作符类型和索引
                    op_type = op[0]
                    op_index = int(op[1:])
                    operators.append((op_type, op_index))

        # 添加元组到列表
        hamiltonian.append((coefficient, operators))

    return hamiltonian



def operator_str_to_tensor(operator, length):
    """
    Convert a given operator string to its tensor representation.
    Args: operator (str): A string representing the operator.
    Example: converting a string like [('Z', 0), ('Z', 1)] to its tensor form
    Returns:]
        Tensors list: The tensors representation of the hamiltonian operators.
        Tensor: The full tensor representation of the hamiltonian term.
    """
    tensors = {
        'Z': tc.tensor([[1, 0], [0, -1]], dtype=tc.complex64),
        'X': tc.tensor([[0, 1], [1, 0]], dtype=tc.complex64),
        'Y': tc.tensor([[0, -1j], [1j, 0]], dtype=tc.complex64),
        'I': tc.eye(2, dtype=tc.complex64)  # Identity matrix
    }

    # Parse the operator string
    # terms = operator.split()

    # Create the list of tensors for each qubit
    all_tensors = [tensors['I']] * length  # Initialize with Identity matrices

    # Fill the tensors list with actual operators
    for op in operator:
        gate = op[0]  # The gate type (X, Y, Z)
        qubit = op[1]  # The qubit index (0, 1, 2, ...)
        all_tensors[qubit] = tensors[gate]  # Update the tensor for the corresponding qubit

    # Compute the full tensor product
    term_tensor = all_tensors[0]
    for t in all_tensors[1:]:
        term_tensor = tc.kron(term_tensor, t)

    # term_tensor = term_tensor.reshape([2] * (2 * length))
    return all_tensors, term_tensor


def pauli_operators(which=None, device='cpu', if_list=False):
    op = {
        'X': tc.tensor([[0.0, 1.0], [1.0, 0.0]], device=device, dtype=tc.float64),
        'Y': tc.tensor([[0.0, -1.0j], [1.0j, 0.0]], device=device, dtype=tc.complex128),
        'Z': tc.tensor([[1.0, 0.0], [0.0, -1.0]], device=device, dtype=tc.float64),
        'I': tc.eye(2).to(device=device, dtype=tc.float64)
    }
    if which is None:
        if if_list:
            return [op['I'], op['X'], op['Y'], op['Z']]
        else:
            return op
    else:
        return op[which]

def choose_device(n=0):
    if n == 'cpu':
        return 'cpu'
    else:
        if tc.cuda.is_available():
            if n is None:
                return tc.device("cuda:0")
            elif type(n) is int:
                return tc.device("cuda:"+str(n))
            else:
                return tc.device("cuda"+str(n)[4:])
        else:
            return tc.device("cpu")




def CNOT():
    return tc.tensor([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]])


def Z_gate():
    return tc.tensor([[1, 0], [0, -1]])


def Y_gate():
    return tc.tensor([[0, -1j], [1j, 0]])


def RZ_gate(theta):
    return tc.linalg.matrix_exp(Z_gate().to(device=theta.device) * 2 * tc.pi * theta * -1j)


def RY_gate(theta):
    return tc.linalg.matrix_exp(Y_gate().to(device=theta.device) * 2 * tc.pi * theta * -1j)


def U(theta_list):
    circ = RZ_gate(theta_list[0])
    circ = tc.matmul(circ, RY_gate(theta_list[1]))
    circ = tc.matmul(circ, RZ_gate(theta_list[2]))
    return circ

def fprint(content, file=None, print_screen=True, append=True):
    if file is None:
        print(content)
    else:
        if append:
            way = 'ab'
        else:
            way = 'wb'
        with open(file, way, buffering=0) as log:
            log.write((content + '\n').encode(encoding='utf-8'))
        if print_screen:
            print(content)
def print_dict(a, keys=None, welcome='', style_sep=': ', end='\n', file=None, print_screen=True, append=True):
    express = welcome
    if keys is None:
        for n in a:
            express += n + style_sep + str(a[n]) + end
    else:
        if type(keys) is str:
            express += keys.capitalize() + style_sep + str(a[keys])
        else:
            for n in keys:
                express += n.capitalize() + style_sep + str(a[n])
                if n is not keys[-1]:
                    express += end
    fprint(express, file, print_screen, append)
    return express


def train_mps_basic_state(iter=60000, para=None, paraMPS=None, target_energy=-2.16,
                          H="""-0.3315 [] + 
                           0.0056 [Y0 Z1 Z2 Z3 Y4] + 
                           0.1814 [Z0]""", mol_qubits=8):
    if para is None:
        para = dict()
    para0 = dict()
    para0['lr'] = 0.01  # learning rate
    para0['lr_decay'] = 0.9  # decaying of learning rate
    para0['sweepTime'] = 100  # sweep time

    para0['isSave'] = True  # whether save MPS
    para0['save_dir'] = './'  # where to save MPS
    para0['save_name'] = 'GMPSdata'  # saving file name
    para0['save_dt'] = 5  # the frequency to save
    para0['record'] = 'record.log'  # name of log file
    para0['device'] = choose_device('cpu')  # cpu or gpu
    para0['dtype'] = tc.float64
    para = dict(para0, **para)

    if paraMPS is None:
        paraMPS = dict()
    paraMPS0 = {
        'length': mol_qubits,
        'd': 2,
        'chi':4,
        'dc': 4,
        'boundary': 'open',
        'spin': False,
        'feature_map': 'cossin',
        'eps': 1e-14,
        'theta': 1.0,
        'device': para['device'],
        'dtype': para['dtype']
    }
    paraMPS = dict(paraMPS0, **paraMPS)

    mps = MPS_basic(para=paraMPS)

    H = parse_hamiltonian(H)
    H_tensor = tc.zeros(((2 ** mps.para['length']) ,(2 ** mps.para['length'])), dtype=tc.complex64)
    for coefficient, operator in H[1:]:
        _, H_tensor_i = operator_str_to_tensor(operator, mps.length)   # operator_tensor.shape=(2**length,2**length)
        H_tensor += coefficient * H_tensor_i

    # 启用异常检测以帮助调试
    tc.autograd.set_detect_anomaly(True)

    # 优化器
    # optimizer = tc.optim.Adam(mps.tensors, lr=para['lr'])
    optimizer = tc.optim.SGD(mps.tensors, lr=0.01, momentum=0.9)

    loss_list = []
    energy_list = []

    # 优化步骤
    for step in range(iter):
        mps.center = -1
        mps.orthogonalized_tensors = [x.clone() for x in mps.tensors]
        mps.center_orthogonalization(
            0, way='svd', dc=mps.dc, normalize=True, is_tensors_leaf=mps.tensors[0].is_leaf)
        for nt in range(mps.length-1, -1, -1):
            mps.center = nt
            mps.normalize_central_tensor(normalize=True, is_tensors_leaf=True)

        psi = full_tensor(mps.orthogonalized_tensors)
        psi = psi / psi.norm()
        psi = psi.flatten()

        optimizer.zero_grad()

        energy = tc.einsum('i,ij,j->', psi.conj().to(dtype=H_tensor.dtype), H_tensor.to(device=psi.device), psi.to(dtype=H_tensor.dtype))
        energy = energy.real + H[0][0]
        loss = abs(energy - target_energy)
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_list.append(loss.item())
        energy_list.append(energy.item())

        if step % 50 == 0:
            print(f"Step {step}, Energy: {energy.item()}, Loss: {loss.item()}")

        if loss.item() < 1e-6:
            break

    trained_mps_tensors = [x.clone() for x in mps.orthogonalized_tensors]
    mol_of_mps_tensors = [x.clone().norm() for x in trained_mps_tensors]
    target = [tensor[:, 0, :] for tensor in trained_mps_tensors]
    s = mps.orthogonalized_tensors[0].shape
    target[0] = tc.matmul(tc.eye(s[2], s[0], device=target[0].device, dtype=target[0].dtype), target[0])
    target[0] = target[0] / target[0].norm()
    s = mps.orthogonalized_tensors[-1].shape
    target[-1] = tc.matmul(target[-1], tc.eye(s[2], s[0], device=target[-1].device, dtype=target[-1].dtype))
    target[-1] = target[-1] / target[-1].norm()
    s = mps.orthogonalized_tensors[-2].shape
    target[-2] = tc.matmul(target[-2], tc.eye(s[2], s[0], device=target[-1].device, dtype=target[-2].dtype))
    target[-2] = target[-2] / target[-2].norm()
    return loss_list, energy_list, target


def train_single_tensor_params(input, target, iter=300, single_qubit=False):
    input = 1/(1+tc.exp(-input))
    input = tc.tensor(input, requires_grad=True, dtype=tc.float64, device=target.device)
    # print("init input:", input)
    loss = nn.MSELoss()
    optimizer = tc.optim.Adam([input], lr=0.01)
    for i in range(iter):
        if single_qubit:
            circuit = U(input)
        else:
            circuit = tc.kron(U(input[0:3]), U(input[3:6]))
            circuit = tc.matmul(circuit, CNOT().to(dtype=circuit.dtype, device=input[0].device))
            circuit = tc.matmul(circuit, tc.kron(U(input[6:9]), U(input[9:12])))

        optimizer.zero_grad()
        output = loss(circuit.real,target[0].to(dtype=circuit.dtype).real) + loss(circuit.imag, target[0].to(dtype=circuit.dtype).imag)
        output.backward(retain_graph=True)
        optimizer.step()

        # if i % 100 == 0:
            # print(f"Epoch {i}, Loss: {output.item()}")
            # print("Gradients:")
            # if input.grad is not None:
            #     input_grad = input.grad.mean().item()
            #     print(f"Theta Gradient: {input_grad}")
            # else:
            #     print("No gradients computed for theta.")
    return input


def mps2circparams(trained_mps_tensors=None, iter=501, filename=None):
    trained_mps_tensors = tc.load(filename)
    trained_params = []
    for n in range(len(trained_mps_tensors)-1):
        trained_tensor_n_params = train_single_tensor_params(tc.randn(12, requires_grad=True, dtype=tc.float64, device=trained_mps_tensors[1].device),
                              tc.tensor(trained_mps_tensors[n], requires_grad=False), iter=iter, single_qubit=False)
        trained_params.append(trained_tensor_n_params)
        print(f"Trained Parameters for tensor {n+1}:", trained_tensor_n_params)
    trained_last_tensor_params = train_single_tensor_params(tc.randn(3, requires_grad=True, dtype=tc.float64, device=trained_mps_tensors[1].device),
                                tc.tensor(trained_mps_tensors[-1], requires_grad=False), iter=iter, single_qubit=True)
    trained_params.append(trained_last_tensor_params)
    return trained_params




