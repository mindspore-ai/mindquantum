from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum import *
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.ma import masked_array
import json
import os
import warnings

"""ignore Warning"""
warnings.filterwarnings('ignore', message='Warning: converting a masked element to nan.')
warnings.filterwarnings('ignore', category=UserWarning, message='findfont: Font family.*not found.')


"""share functions"""
def get_mole_info(geometry:list, fci=True):
    """Complete classical computing tasks based on OpenFermion to obtain molecular information,
        such as FCI energy, HF energy, system Hamiltonian, etc.
        Inputs:
            geometry (list): Geometric structure of the molecule
            run_fci (bool): Whether to run FCI calculation
        Returns:
            hf_energy (float): HF energy
            fci_energy (float): FCI energy
            ham (CI_Hamiltonian): Hamiltonian of the system
            n_qubits (int): Number of qubits
            n_electrons (int): Number of electrons
        """
    basis = "sto3g"
    spin = 0
    mole = MolecularData(geometry, basis, multiplicity=2 * spin + 1)
    mole = run_pyscf(mole, run_fci=fci)
    ham_of = mole.get_molecular_hamiltonian()
    int_ops = InteractionOperator(*(ham_of.n_body_tensors.values()))
    f_ham = FermionOperator(int_ops).compress().normal_ordered()
    return mole.hf_energy, mole.fci_energy, mqchem.CIHamiltonian(f_ham), mole.n_qubits, mole.n_electrons

def fop_to_excitation(fop:FermionOperator):
    """Convert the fermionic excitation operator to tuple form 5^ 4^ 1 0 ->  (5,4,1,0)"""
    return tuple([op for op in next(iter(fop.terms.keys()))])

def fop_to_core(fop:FermionOperator):
    """s_0 [5^ 4] -> (5, 4)"""
    excitation = fop_to_excitation(fop)
    return tuple([op[0] for op in excitation])

def core_to_excitation(core:tuple):
    """(5, 4, 3, 2) -> ((5,1), (4,1), (3,0), (2,0))"""
    return tuple([(core[i], int(i<len(core)/2)) for i in range(len(core))])

def get_cores_pool(f_ham, n_qubits, n_electrons):
    """Obtain the excitation operator pool based on HiUCCSD"""
    ham_terms = set(f_ham.terms.keys())
    uccsd = uccsd_singlet_generator(n_qubits, n_electrons, False)
    pool = []
    for fop in uccsd:
        if fop_to_excitation(fop) in ham_terms:
            pool.append(fop_to_core(fop))
    return pool

def func(x, grad_ops, num_fops):
    """General optimization process of VQE"""
    global measure_counter
    measure_counter.increase(2 * num_fops) # The parameter shift rule requires calculating the expectation value twice
    res = grad_ops(x)
    return res

def run_vqe(sim:mqchem.MQChemSimulator, 
            ham:mqchem.CIHamiltonian, 
            circ:Circuit, 
            num_fops:int,
            p0=None):
    """Run the VQE algorithm"""
    global measure_counter
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    if p0 is None:
        p0 = np.array([0.]*len(circ.params_name))
    res = minimize(func, p0, (grad_ops, num_fops), method='bfgs', jac=True, tol=1e-8)
    return res.fun, res.x, measure_counter.count


"""Ansatz"""
class ADAPT_Ansatz(list):
    """ADAPT-specific ansatz"""
    def __init__(self, ansatz:None|FermionOperator=None):
        super().__init__()
        self.semi_circuit = Circuit() # Pseudo-circuit only for counting parameter name information, not actually executed
        self.fops = []
        if ansatz is not None:
            for fop in ansatz:
                self.append(fop)
        self.masks = []
        
    def get_circuit(self):
        return Circuit([mqchem.UCCExcitationGate(fop) for fop in self.fops])
    
    @property
    def params_name(self):
        """Parameter names in the ansatz"""
        circ = self.get_circuit()
        return circ.params_name
    
    def __str__(self):
        return '_'*10 + '\n' + '\n'.join([str(fop) for fop in self.fops]) + "\n" + '='*10
        
    def append(self, operator: FermionOperator):
        self.fops.append(operator)

    def __iadd__(self, fop:FermionOperator): #type:ignore
        """+="""
        self.append(fop)
        return self



"""Conventional ADAPT-VQE"""
def get_com_mapping(f_ham, cores_pool):
    """Commutator used to calculate the gradient when the excitation operator parameter is 0"""
    mapping = {}
    for core in cores_pool:
        com = (FermionOperator(core_to_excitation(core))*f_ham - f_ham*FermionOperator(core_to_excitation(core))).normal_ordered()
        mapping[core] = com
    return mapping

def get_init_grads(sim:mqchem.MQChemSimulator, 
                    com_mapping:FermionOperator):
    """Calculate the initial gradient magnitudes of each operator in the operator pool for the current state"""
    init_grads = {}
    global measure_counter
    for core, com in com_mapping.items():
        measure_counter.increase(len(com))
        init_grads[core] = abs(sim.get_expectation(mqchem.CIHamiltonian(com)))
    return init_grads

def get_best_core_grad(grads:dict):
    """Select the excitation operator with the largest gradient value"""
    sortd_cores = sorted(grads, key=grads.get, reverse=True)
    best_core = sortd_cores[0]
    norm = np.sqrt(sum([i**2 for i in list(grads.values())]))
    return best_core, norm

def run_adapt_vqe(geometry:list, max_iters:int|None=None, 
                    tol:float=1e-3):
    """Run the ADAPT-VQE algorithm"""
    hf, fci, ham, n_qubits, n_electrons = get_mole_info(geometry, fci=True)
    f_ham = ham.fermion_hamiltonian
    cores_pool = get_cores_pool(f_ham, n_qubits, n_electrons)
    com_mapping = get_com_mapping(f_ham, cores_pool)
    ham_size = len(f_ham) - 1
    theta = PRGenerator("theta_")
    pre_res = {}
    sim = mqchem.MQChemSimulator(n_qubits, n_electrons)
    energys, errors = [hf], [abs(hf - fci)]
    ansatz = ADAPT_Ansatz()
    iter = 0
    mcounts = [0]
    params = np.array([0.])
    init_params = []
    init_measure_counter()
    while True:
        iter += 1
        # print("iter", iter)
        if (max_iters is not None) and (iter > max_iters):
            break
        sim.reset()
        circ = ansatz.get_circuit()
        sim.apply_circuit(circ, pre_res)
        init_grads = get_init_grads(sim, com_mapping)
        # grads = np.array([init_grads[core] for core in cores_pool])
        # print("cores")
        # print(cores_pool)
        # print("grads")
        # print(abs(grads).tolist())
        
        best_core, norm = get_best_core_grad(init_grads)
        if abs(norm) < tol:
            break
        # print("best_core", best_core)
        ansatz += FermionOperator(core_to_excitation(best_core), theta.new())
        circ = ansatz.get_circuit()
        p0 = np.array([pre_res[k] if k in pre_res else 0. for k in circ.params_name]) # warm-start
        sim.reset()
        energy, params, mcount = run_vqe(sim, ham, circ, len(p0)*ham_size, p0)
        init_params.append(abs(params[-1]))
        pre_res = dict(zip(circ.params_name, params))
        energys.append(energy)
        errors.append(abs(fci - energy))
        # print("error", abs(fci - energy))
        # print("*"*20)
        mcounts.append(mcount)
    return energys, errors, params, mcounts, init_params



"""param ADAPT-VQE"""
def get_cores_params(sim:mqchem.MQChemSimulator,
                   sub_hams:dict,
                   sub_hams_size:dict):
    """Optimize the latest parameters of VQE and record the absolute values of the parameters"""
    tmp_params = {}
    for core, sub_ham in sub_hams.items():
        fop = FermionOperator(core_to_excitation(core), 'a')
        tmp_circ = Circuit([mqchem.UCCExcitationGate(fop)])
        sub_ham_size = sub_hams_size[core]
        _, param, mcount = run_vqe(sim, sub_ham, tmp_circ, sub_ham_size)
        tmp_params[core] = param[0]
    return tmp_params, mcount

def get_best_core_param(params:dict):
    """Select the excitation operator with the largest gradient value"""
    abs_params = {core:abs(v) for core, v in params.items()}
    sortd_cores = sorted(abs_params, key=abs_params.get, reverse=True)
    best_core = sortd_cores[0]
    return best_core, params[best_core]

def get_sub_ham(f_ham:FermionOperator, core):
    """Collect core-related sub-Hamiltonians"""
    sub_ham = FermionOperator()
    for fop in f_ham:
        ham_core = fop_to_core(fop)
        if set(core) & set(ham_core):
            sub_ham += fop
    return sub_ham

def get_sub_hams(f_ham, cores_pool):
    """Collect all core-involved sub-Hamiltonians in cores_pool"""
    sub_hams, sub_hams_size = {}, {}
    for core in cores_pool:
        sub_ham = get_sub_ham(f_ham, core)
        sub_hams[core] = mqchem.CIHamiltonian(sub_ham)
        sub_hams_size[core] = len(sub_ham)
    return sub_hams, sub_hams_size

class Study:
    """Class for performing computational tasks"""
    def __init__(self, geo, max_iter:int|None, tol=1e-3, period=1):
        self.max_iter = max_iter
        self.tol = tol
        self.period = period
        
        hf, fci, ham, n_qubits, n_electrons = get_mole_info(geo, fci=True)
        self.hf = hf
        self.fci = fci
        self.energys, self.errors = [self.hf], [abs(self.hf - self.fci)]
        self.ham = ham
        self.f_ham = ham.fermion_hamiltonian
        self.ham_size = len(self.f_ham) - 1
        self.cores_pool = get_cores_pool(self.f_ham, n_qubits, n_electrons)
        self.sub_hams, self.sub_hams_size = get_sub_hams(self.f_ham, self.cores_pool)

        self.sim = mqchem.MQChemSimulator(n_qubits, n_electrons)
        
        self.ansatz = ADAPT_Ansatz()
        self.theta = PRGenerator("theta_")
        self.pre_res = {}
        self.iter = 0
        self.all = False
        self.n_local = 1
        self.mcounts = [0]
        self.params = np.array([0.])
        self.init_params = []
    

    def run(self):
        """Main program"""
        while True:
            self.iter += 1
            if (self.max_iter is not None) and (self.iter > self.max_iter):
                break
            # print("iter", self.iter)
            self.sim.reset()
            circ = self.ansatz.get_circuit()
            self.sim.apply_circuit(circ, self.pre_res)
            # Explore the local parameters of all excitation operators
            cores_params, mcount = get_cores_params(self.sim, self.sub_hams, self.sub_hams_size)
            # print("cores")
            # paramss = np.array([cores_params[core] for core in self.cores_pool])
            # print(self.cores_pool)
            # print("paramss")
            # print(abs(paramss).tolist())
            best_core, param = get_best_core_param(cores_params)
            # print("best_core", best_core)
            
            if abs(param) < self.tol:
                break

            # Add the excitation operator with the largest local parameter to the ansatz
            fop = FermionOperator(core_to_excitation(best_core), self.theta.new())
            self.ansatz += fop
            self.pre_res[fop.params_name[0]] = param
            self.mcounts.append(mcount)

            # Calculate the expectation value without counting the measurement cost, as this is unnecessary
            if self.n_local < self.period:
                fop_circ = Circuit([mqchem.UCCExcitationGate(fop)])
                self.sim.apply_circuit(fop_circ, pr=param)
                energy = self.sim.get_expectation(self.ham)
                self.n_local += 1
                self.all = False

            else:
                energy = self.global_optimizing()

            self.energys.append(energy)
            self.errors.append(abs(energy- self.fci))
            print("error", abs(self.fci - energy))
            print("*"*20)

        energy = self.final_check() # Perform a global optimization before termination
        if energy is not None:
            self.energys[-1] = energy
            self.errors[-1] = abs(energy - self.fci)

        self.params = [self.pre_res[k] for k in self.ansatz.params_name]


    def global_optimizing(self):
        """Global optimization: optimize all parameters in the ansatz."""
        circ = self.ansatz.get_circuit()
        p0 = np.array([self.pre_res[k] for k in circ.params_name]) # hot-start
        self.sim.reset()
        n_fops = self.ham_size * len(self.ansatz.params_name)
        energy, params, mcount = run_vqe(self.sim, self.ham, circ, n_fops, p0)
        self.init_params.append(abs(params[-1]))
        self.pre_res = dict(zip(circ.params_name, params))
        self.n_local = 1
        self.all = True
        self.mcounts[-1] = mcount
        return energy

    def final_check(self):
        """Perform a final check and conduct a global optimization if it has not been done yet."""
        if not self.all:
           energy = self.global_optimizing()
        else:
            energy = None
        return energy

def run_adapt_vqe_param(geo:list, 
                        max_iter:int|None=None, 
                        tol:float=1e-3):
    """Run the ADAPT-VQE algorithm"""
    init_measure_counter()
    study = Study(geo, max_iter, tol)
    study.run()
    return study.energys, study.errors, study.params, study.mcounts



"""Count the number of measurements"""
class MeasureCounter:
    """Used to count the number of measured excitation terms"""
    def __init__(self):
        self.count = 0
    
    def increase(self, num_ham_fops:int):
        self.count += num_ham_fops
    
    def decrease(self, num_ham_fops:int):
        self.count -= num_ham_fops

def init_measure_counter():
    global measure_counter
    measure_counter = MeasureCounter()



"""Construct the molecule"""
def get_bound_list(bound_range:list, num_points:int=21):
    """Obtain the bond length list"""
    return [bound_range[0] + i*(bound_range[1]-bound_range[0])/(num_points-1) for i in range(num_points)]

def get_equilibrium(dist_list, func):
    """Calculate the equilibrium bond length and preview the potential energy curve"""
    fci_list = []
    hf_list = []
    for dist in dist_list:
        geo = func(dist)
        hf, fci, _, _, _ = get_mole_info(geo, fci=True)
        fci_list.append(fci)
        hf_list.append(hf)
    print("Equilibrium bond length: ", dist_list[np.argmin(fci_list)])

    plt.plot(dist_list, fci_list)
    plt.plot(dist_list, hf_list)
    plt.show()
    return fci_list, hf_list

def get_Hs_geometry(dist, n:int=4):
    """H chain molecule"""
    geometry = [['H', [0.0, 0.0, i * dist]] for i in range(n)]
    return geometry

def get_H4_geometry(dist=0.89):
    """H4 """
    geometry = [['H', [0.0, 0.0, i * dist]] for i in range(4)]
    return geometry

def get_H6_geometry(dist=0.92):
    """H6"""
    geometry = [['H', [0.0, 0.0, i * dist]] for i in range(6)]
    return geometry

def get_HF_geometry(dist=1.00):
    """HF"""
    geometry = [['H', [0.0, 0.0, 0.0 * dist]],
                ['F', [0.0, 0.0, 1.0 * dist]]]
    return geometry

def get_LiH_geometry(dist=1.55):
    """LiH"""
    geometry = [['H', [0.0, 0.0, 0.0 * dist]],
                ['Li', [0.0, 0.0, 1.0 * dist]]]
    return geometry

def get_H2O_geometry(dist=1.02):
    """H2O (water) molecule"""
    angle = np.radians(104.5)
    geometry = [['O', [0.0, 0.0, 0.0]],  # Oxygen atom at the origin
                ['H', [dist, 0.0, 0.0]],  # First hydrogen atom on the x-axis
                ['H', [dist * np.cos(angle), dist * np.sin(angle), 0.0]]]
    return geometry

def get_BeH2_geometry(dist=1.32):
    """BeH2 molecule"""
    geometry = [['Be', [0.0, 0.0, 0.0]],  # Beryllium atom at the origin
                ['H', [0.0, 0.0, dist]],  # First hydrogen atom on the z-axis
                ['H', [0.0, 0.0, -dist]]]  # Second hydrogen atom on the negative z-axis
    return geometry

def get_N2_geometry(dist=1.2):
    """N2 molecule"""  # 补充缺失的注释
    geometry = [['N', [0.0, 0.0, 0.0]],
                ['N', [0.0, 0.0, dist]]]
    return geometry

def get_CO_geometry(dist=1.2):
    """Generate the Cartesian coordinate configuration of CO molecule (unit: Å), default distribution along the z-axis"""
    # Carbon atom at the origin, oxygen atom on the z-axis
    geometry = [
        ['C', [0.0, 0.0, 0.0]],
        ['O', [0.0, 0.0, dist]]
    ]
    return geometry

def get_NH3_geometry(dist=1.09, angle=107.3):
    """
    Generate the geometric configuration of NH3 molecule (triangular pyramidal structure)
    """
    # Nitrogen atom at the coordinate origin (C₃ axis along the z-axis)
    N = ['N', [0.0, 0.0, 0.0]]
    
    # Calculate H atom coordinates: triangular pyramidal geometry based on bond length and bond angle
    theta = np.radians(angle)  # Convert to radians
    # Projection radius of H atoms on the xy-plane & z-axis coordinate (height of the triangular pyramid)
    r_xy = dist * np.sin(theta / 2)
    z_H = -dist * np.cos(theta / 2)  # H atoms are located below the N atom
    
    # Three H atoms are evenly distributed in the xy-plane (120° apart, satisfying C₃ symmetry)
    H1 = ['H', [r_xy, 0.0, z_H]]  # Along the x-axis
    H2 = ['H', [r_xy * np.cos(2*np.pi/3), r_xy * np.sin(2*np.pi/3), z_H]]  # 120°
    H3 = ['H', [r_xy * np.cos(4*np.pi/3), r_xy * np.sin(4*np.pi/3), z_H]]  # 240°
    
    geometry = [N, H1, H2, H3]
    return geometry

def get_CH4_geometry(dist=1.09):
    """Generate the coordinates of the regular tetrahedral geometric structure of CH4 molecule"""
    # Carbon atom at the origin
    C = np.array([0.0, 0.0, 0.0])
    
    # Coordinates of H atoms in regular tetrahedron (bond angle ~109.5°)
    # Calculate regular tetrahedron vertex coordinates: based on cube vertices
    h1 = np.array([1, 1, 1])
    h2 = np.array([1, -1, -1])
    h3 = np.array([-1, 1, -1])
    h4 = np.array([-1, -1, 1])
    
    # Normalize and multiply by bond length
    h1 = dist * h1 / np.linalg.norm(h1)
    h2 = dist * h2 / np.linalg.norm(h2)
    h3 = dist * h3 / np.linalg.norm(h3)
    h4 = dist * h4 / np.linalg.norm(h4)
    
    return [['C', C], ['H', h1], ['H', h2], ['H', h3], ['H', h4]]

def get_NaH_geometry(dist=1.64):
    """NaH molecule"""
    geometry = [['H', [0.0, 0.0, 0.0 * dist]],
                ['Na', [0.0, 0.0, 1.0 * dist]]]
    return geometry

def get_C2H4_geometry(d1=1.34, d2=1.08, a=120.0):
    """Generate the geometric configuration of C2H4 (ethylene) molecule"""
    rad = a / 180 * np.pi
    x, y = d2 * np.cos(rad), d2 * np.sin(rad)
    hc = d1 / 2
    return [
        ['C', [-hc, 0, 0]], ['C', [hc, 0, 0]],
        ['H', [-hc - x, y, 0]], ['H', [-hc - x, -y, 0]],
        ['H', [hc + x, y, 0]], ['H', [hc + x, -y, 0]]
    ]


"""Save the molecular structure"""
def save_meta_data(file_path, dists, geos):
    if not os.path.exists(file_path):
        target = {}
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                target = {}
            else:
                try:
                    target = json.loads(content)
                except json.JSONDecodeError:
                    print(f"Warning: {file_path} is not a valid JSON file, will be initialized as an empty dictionary")
                    target = {}
    
    target["dists"] = dists
    target["geos"] = geos

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False, indent=4)
    print(f"The data has been saved to {file_path}")


"""Plot"""
plt.rcParams.update({
    'text.usetex':True,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'SimSun'],
    'font.size': 10,                   
    'axes.linewidth': 0.8,             
    'xtick.major.width': 0.8,          
    'ytick.major.width': 0.8,         
    'grid.linewidth': 0.5,             
    'grid.linestyle': '--',            
    'legend.frameon': False,           
    'legend.loc': 'best',              
    'legend.fontsize': 9,              
    'mathtext.fontset': 'cm',          
    'axes.unicode_minus': False,       
    'figure.dpi': 300,                
    'savefig.dpi': 300,               
    'savefig.format': 'pdf',          
    'xtick.labelsize': 14,         
    'ytick.labelsize': 14         
})

def draw_errors(errors_grads, errors_params, save_path=None, title="Error Comparison", 
                xlabel="Iteration", ylabel=r"Absolute Error"):
    errors_grads_list = list(errors_grads)
    errors_params_list = list(errors_params)
    
    length = max(len(errors_grads_list), len(errors_params_list))
    errors_grads_list += [None] * (length - len(errors_grads_list))
    errors_params_list += [None] * (length - len(errors_params_list))

    def to_masked_array(lst):
        arr = np.array(lst, dtype=float) 
        return masked_array(arr, np.isnan(arr)) 
    
    errors_grads_masked = to_masked_array(errors_grads_list)
    errors_params_masked = to_masked_array(errors_params_list)
    
    xs = np.arange(length)
    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
    ax.plot(xs, errors_grads_masked, label='other', color='#2ca02c', 
            linewidth=1.5, linestyle='--', marker='', markersize=4)
    ax.plot(xs, errors_params_masked, label='our', color='#1f77b4', 
            linewidth=2, linestyle=':', marker='', markersize=4)
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xlim(-0.5, length - 0.5)
    ax.legend(frameon=False, loc='upper right', framealpha=0)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"The chart has been saved to: {save_path}")
    plt.show()

def draw_mcounts(mcounts_grads, mcounts_params, save_path=None, title="Mcounts Comparison", 
                xlabel="Iteration", ylabel=r"Absolute Error"):
    mcounts_grads_list = list(mcounts_grads)
    mcounts_params_list = list(mcounts_params)
    
    length = max(len(mcounts_grads_list), len(mcounts_params_list))
    mcounts_grads_list += [None] * (length - len(mcounts_grads_list))
    mcounts_params_list += [None] * (length - len(mcounts_params_list))

    def to_masked_array(lst):
        arr = np.array(lst, dtype=float) 
        return masked_array(arr, np.isnan(arr)) 
    
    errors_grads_masked = to_masked_array(mcounts_grads_list)
    errors_params_masked = to_masked_array(mcounts_params_list)
    
    xs = np.arange(length)
    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
    ax.plot(xs, errors_grads_masked, label='other', color='#2ca02c', 
            linewidth=1.5, linestyle='--', marker='', markersize=4)
    ax.plot(xs, errors_params_masked, label='our', color='#1f77b4', 
            linewidth=2, linestyle=':', marker='', markersize=4)
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xlim(-0.5, length - 0.5)
    ax.legend(frameon=False, loc='upper left', framealpha=0)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"The chart has been saved to: {save_path}")
    plt.show()

def draw_params(params_grads, params_params, save_path=None, dpi=300):
    params_grads_abs = list(np.abs(params_grads))
    params_params_abs = list(np.abs(params_params))
    
    length = max(len(params_grads_abs), len(params_params_abs))
    params_grads_abs += [0] * (length - len(params_grads_abs))
    params_params_abs += [0] * (length - len(params_params_abs))

    xs = np.arange(1, length + 1)
    bar_width = 0.7

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, tight_layout=True)
    fig.suptitle('Absolute Values of Two Parameter Sets', fontsize=12, fontweight='bold', y=0.98)

    ax1.bar(
        xs, params_grads_abs,
        width=bar_width,
        color='#2ca02c', 
        edgecolor='black',
        linewidth=0.5,
        alpha=0.9
    )
    ax1.set_ylabel('Absolute Value')
    ax1.set_title('Gradient Parameters', fontweight='bold', pad=8)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    max_grad = np.max(params_grads_abs)
    ax1.set_ylim(0, max_grad * 1.15 if max_grad > 0 else 0.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.bar(
        xs, params_params_abs,
        width=bar_width,
        color='#1f77b4',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.9
    )
    ax2.set_xlabel('Parameter Index')
    ax2.set_ylabel('Absolute Value')
    ax2.set_title('Model Parameters', fontweight='bold', pad=8)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    max_param = np.max(params_params_abs)
    ax2.set_ylim(0, max_param * 1.15 if max_param > 0 else 0.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format=save_path.split('.')[-1])
        print(f"The chart has been saved to: {save_path}")
    plt.show()

def pad_data_to_same_length(data_list):
    """Pad arrays of different lengths to the same length, filling the insufficient parts with NaN"""
    max_len = max([len(data) for data in data_list])
    padded_list = []
    for data in data_list:
        padded = np.pad(data.astype(np.float64), 
                       (0, max_len - len(data)), 
                       constant_values=np.nan)
        padded_list.append(padded)
    return padded_list, max_len

def get_optimal_tick_step(max_value, target_ticks=8):
    """Based on the value range and target tick count, automatically calculate the optimal tick interval"""
    candidates = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    ideal_step = max_value / target_ticks
    for step in candidates:
        if step >= ideal_step:
            return step
    return candidates[-1]

def draw_data_bars(data_0, data_1, title, y_label, vlines_pos=None, target_tick_count=8, save_file=None):
    all_data = [abs(np.array(data_0)), abs(np.array(data_1))]
    padded_data, max_len = pad_data_to_same_length(all_data)
    data_0_padded, data_1_padded = padded_data

    fig, ax = plt.subplots(figsize=(6, 3.8))
    
    x = np.arange(max_len)
    y0 = data_0_padded 
    y1 = -data_1_padded

    bar_width = 0.8
    ax.bar(x, y0, color='#1f77b4', edgecolor='black', alpha=0.8, 
           label='ADAPT-VQE', width=bar_width)
    ax.bar(x, y1, color='#ff7f0e', edgecolor='black', alpha=0.8, 
           label='Param-ADAPT-VQE', width=bar_width)

    if vlines_pos is not None and isinstance(vlines_pos, list):
        for pos in vlines_pos:
            if 0 <= pos < max_len:
                ax.axvline(x=pos, color='#276dc9', linestyle='--', linewidth=1.2, alpha=0.4)

    upper_max = np.nanmax(y0) if len(y0) > 0 else 0
    lower_max = np.nanmax(np.abs(y1)) if len(y1) > 0 else 0
    
    upper_step = get_optimal_tick_step(upper_max, target_tick_count)
    lower_step = get_optimal_tick_step(lower_max, target_tick_count)
    
    upper_scale = np.ceil(upper_max / upper_step) * upper_step
    lower_scale = np.ceil(lower_max / lower_step) * lower_step
    
    upper_ticks = np.arange(0, upper_scale + upper_step, upper_step)
    lower_ticks = np.arange(-lower_scale, 0 + lower_step, lower_step)
    
    y_ticks_pos = np.unique(np.concatenate([lower_ticks, upper_ticks]))
    
    decimal_places = 0 if max(upper_step, lower_step) >= 1.0 else 1
    ax.set_yticks(y_ticks_pos)
    ax.set_yticklabels([f'{abs(tick):.{decimal_places}f}' for tick in y_ticks_pos])

    # ax.set_xlabel('Iteration', fontsize=14)
    ax.set_xlabel('Excitation Operators', fontsize=14)  
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20, y=-0.32) 

    tick_step = max(1, max_len // 10)
    xticks = range(0, max_len + tick_step, tick_step)
    ax.set_xticks(xticks)
    ax.set_xlim(-bar_width/2, max_len - bar_width/2)

    ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.5, which='major', axis='y')
    ax.legend(loc='upper right', fontsize=12)

    plt.subplots_adjust(bottom=0.18)
    plt.tight_layout()
    plt.show()
    if save_file is not None:
        fig.savefig(f"{save_file}.pdf", format="pdf", dpi=300, bbox_inches="tight")


"""Save running data"""
def to_save(file_path:str, method:str, point:int, data_type:str, data):
    """Save data to target, with file_name=H2O, method=grad_adapt/params_adapt_{period}, point=10, data_type=energys, data=list"""
    with open(file_path, "r", encoding="utf-8") as f:
        target = json.load(f)
    
    if method not in target:
        target[method] = {}
    point_ = f"point_{point}" 
    if point_ not in target[method]:
        target[method][point_] = {}
    target[method][point_][data_type] = data

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False, indent=4)
