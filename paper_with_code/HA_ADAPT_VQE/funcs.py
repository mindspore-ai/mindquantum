from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum import *
from mindquantum import mqchem
from mindquantum import FermionOperator, InteractionOperator, uccsd_singlet_generator, Circuit
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.ma import masked_array
import json
import os
import warnings
"""share functions"""

"""ignore Warning"""
warnings.filterwarnings('ignore', message='Warning: converting a masked element to nan.')
warnings.filterwarnings('ignore', category=UserWarning, message='findfont: Font family.*not found.')


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


def get_mole_info(geometry:list, fci=True):
    """Complete classical computing tasks based on OpenFermion to obtain molecular information,
        such as FCI energy, HF energy, system Hamiltonian, etc.
        Inputs:
            geometry (list): Geometric structure of the molecule
            run_fci (bool): Whether to run FCI calculation
        Returns:
            hf_energy (float): HF energy
            fci_energy (float): FCI energy
            f_ham (FermionOperator): Hamiltonian of the system
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
    return mole.hf_energy, mole.fci_energy, f_ham, mole.n_qubits, mole.n_electrons


def fop_to_core(fop:FermionOperator):
    """s_0 [5^ 4] -> (5, 4)"""
    excitation = fop_to_excitation(fop)
    return tuple([op[0] for op in excitation])


def fop_to_excitation(fop:FermionOperator):
    """Convert the fermionic excitation operator to tuple form 5^ 4^ 1 0 ->  (5,4,1,0)"""
    return tuple([op for op in next(iter(fop.terms.keys()))])


def core_to_excitation(core:tuple):
    """(5, 4, 3, 2) -> ((5,1), (4,1), (3,0), (2,0))"""
    return tuple([(core[i], int(i<len(core)/2)) for i in range(len(core))])


def get_cores_pool(f_ham:FermionOperator, n_qubits:int, n_electrons:int):
    """Obtain the excitation operator pool based on HiUCCSD"""
    ham_terms = set(f_ham.terms.keys())
    uccsd = uccsd_singlet_generator(n_qubits, n_electrons, False)
    pool = []
    for fop in uccsd:
        if fop_to_excitation(fop) not in ham_terms: # HIUCCSD
            continue
        pool.append(fop_to_core(fop))
    return pool


def opti_func(x, grad_ops, num_fops):
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
    res = minimize(opti_func, p0, (grad_ops, num_fops), method='bfgs', jac=True)
    return res.fun, res.x, measure_counter.count


def get_cores_com(f_ham:FermionOperator, cores_pool:list):
    """Commutator used to calculate the gradient when the excitation operator parameter is 0"""
    cores_com = {}
    for core in cores_pool:
        com = (FermionOperator(core_to_excitation(core))*f_ham - f_ham*FermionOperator(core_to_excitation(core))).normal_ordered()
        cores_com[core] = com
    return cores_com


def get_cores_init_grad(sim:mqchem.MQChemSimulator, 
                        cores_com:dict):
    """Calculate the initial gradient magnitudes of each operator in the operator pool for the current state"""
    init_grads = {}
    global measure_counter
    for core, com in cores_com.items():
        measure_counter.increase(len(com))
        init_grads[core] = abs(sim.get_expectation(mqchem.CIHamiltonian(com)))
    return init_grads


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


def get_sub_hams_expectation(sim:mqchem.MQChemSimulator, sub_hams:dict, sub_hams_size:dict):
    """Calculate the expectation values of each sub-Hamiltonian"""
    global measure_counter
    measure_counter.increase(2 * sum(sub_hams_size.values())) # Calculate the initial state and the final state respectively once

    sub_hams_expectation = {}
    for core, sub_ham in sub_hams.items():
        sub_hams_expectation[core] = sim.get_expectation(sub_ham)
    return sub_hams_expectation

def get_cores_init_param(sim:mqchem.MQChemSimulator,
                            sub_hams:dict,
                            sub_hams_size:dict):
    """Optimize the latest parameters of VQE and record the absolute values of the parameters"""
    cores_init_param = {}
    cores_init_energy = {}
    for core, sub_ham in sub_hams.items():
        fop = FermionOperator(core_to_excitation(core), 'a')
        tmp_circ = Circuit([mqchem.UCCExcitationGate(fop)])
        sub_energy, param, mcount = run_vqe(sim, sub_ham, tmp_circ, sub_hams_size[core], np.array([0.]))
        cores_init_param[core] = param[0]
        cores_init_energy[core] = sub_energy
    return cores_init_param, cores_init_energy

def get_cores_energy_reduction(sub_hams_expectation:dict, cores_init_energy:dict):
    """Calculate the energy reduction of each core"""
    cores_energy_reduction = {}
    for core, sub_ham_expectation in sub_hams_expectation.items():
        cores_energy_reduction[core] = sub_ham_expectation - cores_init_energy[core]
    return cores_energy_reduction


def get_best_cores(cores_data:dict, num:int=1):
    """Select the excitation operator with the largest gradient value"""
    abs_data = {core:abs(v) for core, v in cores_data.items()}
    sortd_cores = sorted(abs_data, key=abs_data.get, reverse=True)
    best_cores = sortd_cores[:num]
    norm = np.sqrt(sum([i**2 for i in list(abs_data.values())]))
    return best_cores, [cores_data[core] for core in best_cores], norm


class ADAPT_Ansatz(list):
    """ADAPT ansatz"""
    def __init__(self, tol:float=1e-4, n_removes:int=2):
        """tol: Parameter value threshold, n_removes: the maximum number of elements to be removed"""
        super().__init__()
        self.tol = tol
        self.n_removes = n_removes
        self.fops = []
        self.circ = Circuit()
        self.size = 0
        self.params_name = []
        self.cores = []

    def update(self):
        self.circ = Circuit([mqchem.UCCExcitationGate(fop) for fop in self.fops])
        self.params_name = self.circ.params_name
        self.size = len(self.circ)
        self.cores = []
        for fop in self.fops:
            self.cores.append(fop_to_core(fop))
    
    def __str__(self):
        return '_'*10 + '\n' + '\n'.join([str(fop) for fop in self.fops]) + "\n" + '='*10
        
    def append(self, fop:FermionOperator):
        self.fops.append(fop)
        self.update()

    def __iadd__(self, fop:FermionOperator):
        """+="""
        self.append(fop)
        return self

    def check(self, params):
        """Delete n_removes elements whose parameter values in the candidate settings are less than the threshold""" 
        abs_params = abs(params)
        mask_less = abs_params < self.tol
        last_index = max([j for j, v in enumerate(mask_less) if v == False])
        weak_pairs = [(idx, param) for idx, param in enumerate(abs_params[:last_index]) if param < self.tol]
        most_weak_pairs = sorted(weak_pairs, key=lambda x:x[1])
        indices_del = [item[0] for item in most_weak_pairs[:self.n_removes]]
        if indices_del:
            self.fops = [self.fops[i] for i in range(len(self.fops)) if i not in indices_del]
            params = [params[i] for i in range(len(params)) if i not in indices_del]
            self.update()
        return params,  [abs_params[i] for i in indices_del]
    
    def prune(self, params):
        """Pruned-ADAPT-VQE"""
        factors = []
        for i, v in enumerate(params):
            factors.append(1 / ((v+1e-10)**2) * np.exp(-10*(i+1)))
        factors = np.array(factors)
        tau = 0.1 * np.mean(abs(params[-4:]))
        
        if len(params) > 10:
            index = np.argmax(factors)
            if abs(params[index]) < tau:
                self.fops.pop(index)
                params = np.delete(params, index)
                self.update()
        return params


class Study:
    def __init__(self, geo, max_size:int=120, tol=1e-4):
        self.max_size = max_size
        self.tol = tol
        hf, fci, f_ham, n_qubits, n_electrons = get_mole_info(geo, fci=True)
        self.hf = hf
        self.fci = fci
        self.energys, self.errors = [self.hf], [abs(self.hf - self.fci)]
        self.f_ham = f_ham
        self.ham = mqchem.CIHamiltonian(f_ham)
        self.ham_size = len(self.f_ham) - 1
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.cores_pool = get_cores_pool(self.f_ham, n_qubits, n_electrons)
        self.cores_com = get_cores_com(self.f_ham, self.cores_pool)
        self.sub_hams, self.sub_hams_size = get_sub_hams(self.f_ham, self.cores_pool)
        self.cores_weight = get_cores_weight(self.f_ham, self.n_qubits, self.n_electrons)
        self.theta = PRGenerator("theta_")
        self.pre_res = {}
        self.sim = mqchem.MQChemSimulator(n_qubits, n_electrons)
        self.ansatz = ADAPT_Ansatz()
        self.params = np.array([0.]) # Currently for placeholder use only
        self.mcounts = [np.nan] # Initially set to None, no display in plotting
        self.ansatz.size = 0
        self.size_his = [0]
        self.n_candidates = 10
        init_measure_counter()


    def global_optimizing(self, new=True):
        """Global optimization: optimize all parameters in the ansatz."""
        circ = self.ansatz.circ
        p0 = np.array([self.pre_res[k] for k in circ.params_name]) # hot-start
        self.sim.reset()
        energy, params, mcount = run_vqe(self.sim, self.ham, circ, self.ham_size*len(p0), p0)
        self.pre_res = dict(zip(circ.params_name, params))
        self.params = params
        if new:
            self.energys.append(energy)
            self.errors.append(abs(energy- self.fci))
            self.mcounts.append(mcount)
            self.size_his.append(self.ansatz.size)
            
        else:
            self.energys[-1] = energy
            self.errors[-1] = abs(energy- self.fci)
            self.mcounts[-1] = mcount
            self.size_his[-1] = self.ansatz.size


class ADAPT_Study(Study):
    """Standard ADAPT-VQE"""
    def run(self):
        global measure_counter
        while True:
            if self.ansatz.size > self.max_size:
                break
            self.sim.reset()
            self.sim.apply_circuit(self.ansatz.circ, self.pre_res)
            cores_init_grad = get_cores_init_grad(self.sim, self.cores_com)
            best_cores, best_grads, grads_norm = get_best_cores(cores_init_grad, num=1)
            if abs(grads_norm) < self.tol:
                break
            fop = FermionOperator(core_to_excitation(best_cores[0]), self.theta.new())
            self.ansatz += fop
            self.pre_res[fop.params_name[0]] = 0
            self.global_optimizing(new=True)

def run_adapt_vqe(geo:list, 
                    max_size:int=120, 
                    tol:float=1e-4):
    """Run the ADAPT-VQE algorithm"""
    study = ADAPT_Study(geo, max_size, tol)
    study.run()
    return study.energys, study.errors, study.params, study.mcounts, study.size_his


class Param_Study(Study):
    """param ADAPT-VQE"""
    def run(self):
        while True:
            if self.ansatz.size > self.max_size:
                break

            self.sim.reset()
            self.sim.apply_circuit(self.ansatz.circ, self.pre_res)
            cores_init_param, _ = get_cores_init_param(self.sim, self.sub_hams, self.sub_hams_size)
            best_cores, best_params, params_norm = get_best_cores(cores_init_param, num=1)
            if abs(params_norm) < self.tol:
                break

            fop = FermionOperator(core_to_excitation(best_cores[0]), self.theta.new())
            self.ansatz += fop
            self.pre_res[fop.params_name[0]] = best_params[0]
            self.global_optimizing(new=True)


def run_adapt_vqe_param(geo:list, 
                        max_size:int=120, 
                        tol:float=1e-4):
    """Run the Param-ADAPT-VQE algorithm"""
    init_measure_counter()
    study = Param_Study(geo, max_size, tol)
    study.run()
    return study.energys, study.errors, study.params, study.mcounts, study.size_his



class Pruned_Study(Study):
    """Pruned-ADAPT-VQE""" 
    def run(self):
        while True:
            if self.ansatz.size > self.max_size:
                break
            self.sim.reset()
            self.sim.apply_circuit(self.ansatz.circ, self.pre_res)
            cores_init_grad = get_cores_init_grad(self.sim, self.cores_com)
            best_cores, best_grads,  grads_norm = get_best_cores(cores_init_grad, num=1)
            if abs(grads_norm) < self.tol:
                break
            fop = FermionOperator(core_to_excitation(best_cores[0]), self.theta.new())
            self.ansatz += fop
            self.pre_res[fop.params_name[0]] = 0
            self.global_optimizing(True)

            params = np.array([self.pre_res[k] for k in self.ansatz.params_name])
            new_params = self.ansatz.prune(params)
            if len(params) > len(new_params):
                self.global_optimizing(new=False)


def run_adapt_vqe_pruned(geo:list,
                         max_size:int=120, 
                         tol:float=1e-4):
    """Run the Param-ADAPT-VQE algorithm"""
    init_measure_counter()
    study = Pruned_Study(geo, max_size, tol)
    study.run()
    return study.energys, study.errors, study.params, study.mcounts, study.size_his



def get_cores_weight(f_ham:FermionOperator, 
                     n_qubits:int, 
                     n_electrons:int):
    """Obtain the excitation operator pool based on HiUCCSD"""
    ham_terms = set(f_ham.terms.keys())
    uccsd = uccsd_singlet_generator(n_qubits, n_electrons, False)
    cores_weight = {}
    for fop in uccsd:
        if fop_to_excitation(fop) not in ham_terms: # HIUCCSD
            continue
        cores_weight[fop_to_core(fop)] = f_ham.get_coeff(fop_to_excitation(fop)).const
    return cores_weight

def consider_weight(cores_weight:dict, 
                    cores_init_param:dict):
    weighted_params = {}
    for core, param in cores_init_param.items():
        weighted_params[core] = abs(0.5*np.sin(2*param) * cores_weight.get(core, 0))
    return weighted_params


class HA_Study(Study):
    """Hamiltonian Aware ADAPT-VQE"""
    def run(self):
        while True:
            if self.ansatz.size > self.max_size:
                break
            self.sim.reset()
            self.sim.apply_circuit(self.ansatz.circ, self.pre_res)
            cores_init_param, _ = get_cores_init_param(self.sim, self.sub_hams, self.sub_hams_size)
            cores_wparams = consider_weight(self.cores_weight, cores_init_param)
            best_cores, best_wparams, wparams_norm = get_best_cores(cores_wparams, num=1)
            norm = np.sqrt(sum([cores_init_param[k]**2 for k in cores_init_param.keys()]))
            if norm < self.tol:
                break
            
            fop = FermionOperator(core_to_excitation(best_cores[0]), self.theta.new())
            self.ansatz += fop
            self.pre_res[fop.params_name[0]] = cores_init_param[best_cores[0]]
            self.global_optimizing(new=True)
            tmp_error = self.errors[-1]
            params, del_params = self.ansatz.check(self.params)
            num_del = len(self.params) - len(params)
            if num_del > 0:
                self.params = params
                self.pre_res = {k:v for k, v in zip(self.ansatz.params_name, params)}
                self.global_optimizing(new=False)
                if (self.errors[-1] - tmp_error) > 1e-7:
                    self.ansatz.tol *= 0.5

def run_adapt_vqe_ha(geo:list,
                    max_iter:int=120,
                    tol:float=1e-4):
    """Run the Hamiltonian Aware ADAPT-VQE algorithm"""
    init_measure_counter()
    study = HA_Study(geo, max_iter, tol)
    study.ansatz.tol = 5e-4
    study.ansatz.n_removes = 100
    study.run()
    return study.energys, study.errors, study.params, study.mcounts, study.size_his


def read_data(mole, point, data_type):
    file_path = f"{mole}.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    grad_adapt = saved_data['grad_adapt'][f"point_{point}"][data_type]
    pruned_adapt = saved_data['pruned_adapt'][f"point_{point}"][data_type]
    param_adapt = saved_data['param_adapt'][f"point_{point}"][data_type]
    ha_adapt = saved_data['ha_adapt'][f"point_{point}"][data_type]
    return grad_adapt, pruned_adapt, param_adapt, ha_adapt


warnings.filterwarnings('ignore', message='Warning: converting a masked element to nan.')
warnings.filterwarnings('ignore', category=UserWarning, message='findfont: Font family.*not found.')

styles = ['--',  '-.', ':', '-']
colors = ['#1f77b4', '#d62728', '#9467bd',  '#ff7f0e']
labels = ['ADAPT-VQE', 'Pruned-ADAPT-VQE', 'Param-ADAPT-VQE', 'HA-ADAPT-VQE']



def safe_array(data):
    arr = []
    for x in data:
        if x is None or x == 'None':
            arr.append(np.nan)
        else:
            arr.append(float(x))
    return np.array(arr, dtype=np.float64)

def set_first_to_nan(data):
    data = safe_array(data)
    if len(data) > 0:
        data[0] = np.nan
    return data

def sort_by_error(errors, mcounts):
    errors = safe_array(errors)
    mcounts = safe_array(mcounts)
    
    valid_mask = ~np.isnan(errors) & ~np.isnan(mcounts)
    valid_errors = errors[valid_mask]
    valid_mcounts = mcounts[valid_mask]
    
    sorted_indices = np.argsort(valid_errors)[::-1]
    sorted_errors = valid_errors[sorted_indices]
    sorted_mcounts = valid_mcounts[sorted_indices]
    
    return sorted_errors, sorted_mcounts


def draw_errors(mole, point):
    errors_list = read_data(mole, point, 'errors')
    size_his_list = read_data(mole, point, 'size_his')

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (errors, size_his) in enumerate(zip(errors_list, size_his_list)):
        plt.plot(size_his, errors, linestyle=styles[i], color=colors[i], linewidth=2, label=labels[i])
    ax.set_yscale('log')
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    plt.close()


def draw_mcounts(mole, point):
    errors_list = read_data(mole, point, 'errors')
    mcounts_list = read_data(mole, point, 'mcounts')
    new_errors_list = []
    new_mcounts_list = []
    for errors, mcounts in zip(errors_list, mcounts_list):
        e, m = sort_by_error(errors, mcounts)
        new_errors_list.append(e)
        new_mcounts_list.append(m)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (errors, mcounts) in enumerate(zip(new_errors_list, new_mcounts_list)):
        plt.plot(errors, mcounts, linestyle=styles[i], color=colors[i], linewidth=2, label=labels[i])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()

    all_errors = np.concatenate([safe_array(e) for e in errors_list])
    valid_errors = all_errors[~np.isnan(all_errors)]
    all_mcounts = np.concatenate([safe_array(m) for m in mcounts_list])
    valid_mcounts = all_mcounts[~np.isnan(all_mcounts)]
    
    if len(valid_errors) > 0:
        x_max = np.max(valid_errors)
        x_min = np.min(valid_errors) * 0.8
        ax.set_xlim(x_max, x_min)
    
    if len(valid_mcounts) > 0:
        y_min = np.min(valid_mcounts) * 0.8
        y_max = np.max(valid_mcounts) * 1.2
        ax.set_ylim(y_min, y_max)
    
    chem_accuracy = 1e-3
    y_min, y_max = ax.get_ylim()
    ax.fill_between(
        x=np.linspace(0, chem_accuracy, 100),
        y1=y_min, y2=y_max,
        color='#add8e6', alpha=0.3,
        label= r'Chemical Precision'
        )

    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, which='major')
    ax.grid(False, which='minor')

    plt.tight_layout()
    plt.show()
    plt.close()


"""Construct the molecule"""
def get_dist_list(bound_range:list, num_points:int=11):
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
    
    return [['C', C.tolist()], ['H', h1.tolist()], ['H', h2.tolist()], ['H', h3.tolist()], ['H', h4.tolist()]]

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
    target = {}
    target["dists"] = dists
    target["geos"] = geos

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False, indent=4)
    print(f"The data has been saved to {file_path}")



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

