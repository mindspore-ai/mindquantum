#!/usr/bin/env python3

from __future__ import annotations
from typing import Any, Union, Dict, List, Iterable, Callable
import os, tempfile, uuid, sys
import numpy as np
from collections import UserDict

from . import data,  stopper, interfaces
from .decorators import doc_inherit

class model():
    '''
    Parent (super) class for models to enable useful features such as logging during geometry optimizations.
    '''
    nthreads = 0
    def set_num_threads(self, nthreads=0):
        # implement for each subclass
        if nthreads:
            self.nthreads = nthreads

    def config_multiprocessing(self):
        '''
        for scripts that need to be executed before running model in parallel
        '''
        pass

    def parse_args(self, args):
        # for command-line arguments parsing
        pass

    def _predict_geomopt(self,
        return_string=False,
        dump_trajectory_interval=None,
        filename=None,
        format='json',
        print_properties=None,
        molecule: data.molecule = None,
        calculate_energy: bool = True, 
        calculate_energy_gradients: bool = True,
        **kwargs):

        self.predict(molecule=molecule, 
                     calculate_energy=calculate_energy,
                     calculate_energy_gradients=calculate_energy_gradients, 
                     **kwargs)

        if dump_trajectory_interval != None:
            opttraj = data.molecular_trajectory()
            opttraj.load(filename=filename, format=format)
            nsteps = len(opttraj.steps)
            if print_properties == 'all' or type(print_properties) == list:
                printstrs = []
                printstrs += [' %s ' % ('-'*78)]
                printstrs += [f' Iteration {nsteps+1}']
                printstrs += [' %s \n' % ('-'*78)]
                printstrs += [molecule.info(properties=print_properties, return_string=True)]
                printstrs = '\n'.join(printstrs) + '\n'
                if not return_string:
                    print(printstrs)
            opttraj.steps.append(data.molecular_trajectory_step(step=nsteps, molecule=molecule))
            opttraj.dump(filename=filename, format=format)
            moldb = data.molecular_database()
            moldb.molecules = [each.molecule for each in opttraj.steps]
            xyzfilename = os.path.splitext(os.path.basename(filename))[0]
            moldb.write_file_with_xyz_coordinates(f'{xyzfilename}.xyz')
        if return_string and (dump_trajectory_interval != None) and (print_properties == 'all' or type(print_properties) == list): return printstrs

    def predict(
        self, 
        molecular_database: data.molecular_database = None, 
        molecule: data.molecule = None,
        calculate_energy: bool = False, 
        calculate_energy_gradients: bool = False, 
        calculate_hessian: bool = False,
        **kwargs,
    ):
        '''
        Make predictions for molecular geometries with the model.

        Arguments:
            molecular_database (:class:`mlatom.data.molecular_database`, optional): A database contains the molecules whose properties need to be predicted by the model.
            molecule (:class:`mlatom.models.molecule`, optional): A molecule object whose property needs to be predicted by the model.
            calculate_energy (bool, optional): Use the model to calculate energy.
            calculate_energy_gradients (bool, optional): Use the model to calculate energy gradients.
            calculate_hessian (bool, optional): Use the model to calculate energy hessian.
        '''
        # for universal control of predicting behavior
        self.set_num_threads()

        if molecular_database != None:
            molecular_database = molecular_database
        elif molecule != None:
            molecular_database = data.molecular_database([molecule])
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            raise ValueError(errmsg)
        return molecular_database
    
    def _call_impl(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    
    __call__ : Callable[..., Any] = _call_impl


class methods(model):
    '''
    Create a model object with a specified method.

    Arguments:
        method (str): Specify the method. Available methods are listed in the section below.
        program (str, optional): Specify the program to use.
        **kwargs: Other method-specific options

    **Available Methods:**

        ``'AIQM1'``, ``'AIQM1@DFT'``, ``'AIQM1@DFT*'``, ``'AM1'``, ``'ANI-1ccx'``, ``'ANI-1x'``, ``'ANI-1x-D4'``, ``'ANI-2x'``, ``'ANI-2x-D4'``, ``'CCSD(T)*/CBS'``, ``'CNDO/2'``, ``'D4'``, ``'DFTB0'``, ``'DFTB2'``, ``'DFTB3'``, ``'GFN2-xTB'``, ``'MINDO/3'``, ``'MNDO'``, ``'MNDO/H'``, ``'MNDO/d'``, ``'MNDO/dH'``, ``'MNDOC'``, ``'ODM2'``, ``'ODM2*'``, ``'ODM3'``, ``'ODM3*'``, ``'OM1'``, ``'OM2'``, ``'OM3'``, ``'PM3'``, ``'PM6'``, ``'RM1'``, ``'SCC-DFTB'``, ``'SCC-DFTB-heats'``.

        Methods listed above can be accepted without specifying a program.
        The required programs still have to be installed though as described in the installation manual.
    
    **Available Programs and Their Corresponding Methods:** 

        .. table::
            :align: center

            ===============  ==========================================================================================================================================================================
            Program          Methods                                                                                                                                                                   
            ===============  ==========================================================================================================================================================================
            TorchANI         ``'AIQM1'``, ``'AIQM1@DFT'``, ``'AIQM1@DFT*'``, ``'ANI-1ccx'``, ``'ANI-1x'``, ``'ANI-1x-D4'``, ``'ANI-2x'``, ``'ANI-2x-D4'``, ``'ANI-1xnr'``                                              
            dftd4            ``'AIQM1'``, ``'AIQM1@DFT'``, ``'ANI-1x-D4'``, ``'ANI-2x-D4'``, ``'D4'``                                                                                                  
            MNDO or Sparrow  ``'AIQM1'``, ``'AIQM1@DFT'``, ``'AIQM1@DFT*'``, ``'MNDO'``, ``'MNDO/d'``, ``'ODM2*'``, ``'ODM3*'``,  ``'OM2'``, ``'OM3'``, ``'PM3'``, ``'SCC-DFTB'``, ``'SCC-DFTB-heats'``
            MNDO             ``'CNDO/2'``, ``'MINDO/3'``, ``'MNDO/H'``, ``'MNDO/dH'``, ``'MNDOC'``, ``'ODM2'``, ``'ODM3'``, ``'OM1'``, semiempirical OMx, DFTB, NDDO-type methods                                                                  
            Sparrow          ``'DFTB0'``, ``'DFTB2'``, ``'DFTB3'``, ``'PM6'``, ``'RM1'``, semiempirical OMx, DFTB, NDDO-type methods                                                                                                              
            xTB              ``'GFN2-xTB'``, semiempirical GFNx-TB methods                                                                                                                                                           
            Orca             ``'CCSD(T)*/CBS'``, DFT                                                                                                                                                      
            Gaussian         ab initio methods, DFT
            PySCF            ab initio methods, DFT
            ===============  ==========================================================================================================================================================================
    
    '''

    methods_map = {
    'aiqm1': ['AIQM1', 'AIQM1@DFT', 'AIQM1@DFT*'],
    'aiqm2': ['AIQM2', 'AIQM2@DFT'],
    'dens': [],
    'torchani': ["ANI-1x", "ANI-1ccx", "ANI-2x", 'ANI-1x-D4', 'ANI-2x-D4', 'ANI-1xnr', 'ANI-1ccx-gelu', 'ANI-1ccx-gelu-D4', 'ANI-1x-gelu', 'ANI-1x-gelu-d4'], 
    'aimnet2': ["AIMNet2@b973c", "AIMNet2@wb97m-d3"],
    'mndo': ['ODM2*', 'ODM2', 'ODM3', 'OM3', 'OM2', 'OM1', 'PM3', 'AM1', 'MNDO/d', 'MNDOC', 'MNDO', 'MINDO/3', 'CNDO/2', 'SCC-DFTB', 'SCC-DFTB-heats', 'MNDO/H', 'MNDO/dH'],
    'sparrow': ['DFTB0', 'DFTB2', 'DFTB3', 'MNDO', 'MNDO/d', 'AM1', 'RM1', 'PM3', 'PM6', 'OM2', 'OM3', 'ODM2*', 'ODM3*', 'AIQM1'],
    'xtb': ['GFN2-xTB'],
    'dftd4': ['D4'],
    'dftd3': ['d3zero', 'd3bj', 'd3bjm', 'd3zerom', 'd3op'],
    'dens': [],
    'ccsdtstarcbs': ['CCSD(T)*/CBS'],
    # in-interface method searching for a empty list
    'pyscf': [],
    'gaussian': [],
    'columbus': [],
    'turbomole': [],
    'orca': [],
    }
    
    def __init__(self, method: str = None, program: str = None, **kwargs):
        # !!! IMPORTANT !!! 
        # It is neccesary to save all the arguments in the model, otherwise it would not be dumped correctly!
        self.method  = method
        self.program = program
        if kwargs != {}: self.kwargs = kwargs
        
        program = self._get_program(method, program)
        self.interface = interfaces.__dict__[program]()(method=method, **kwargs)
    
    @property
    def nthreads(self):
        return self.interface.nthreads
    
    @nthreads.setter
    def nthreads(self, nthreads):
        self.interface.nthreads = nthreads

    def predict(self, *args, **kwargs):
        self.interface.predict(*args, **kwargs)
    
    def config_multiprocessing(self):
        super().config_multiprocessing()
        self.interface.config_multiprocessing()

    @classmethod
    def _get_program(cls, method, program):
        if program:
            if program.lower() not in  ['turbomole', 'columbus'] and not method:
                raise ValueError('A method must be specified')
            if program.casefold() in cls.methods_map:
                return program.casefold()
            else:
                raise ValueError('Unrecognized program')
        else:
            program_list = []
            for program, methods in cls.methods_map.items():
                if methods:
                    if method.casefold() in [m.casefold() for m in methods]:
                        try:
                            interfaces.__dict__[program]()()
                            program_list.append(program)
                        except:
                            pass
                else: 
                    try:
                        if interfaces.__dict__[program]().is_available_method(method):
                            program_list.append(program)  
                    except:
                        pass
                
            if len(set(program_list)) != 0:
                return program_list[0]                     
            raise ValueError("Cannot find appropriate program for the requested method")

    @classmethod
    def known_methods(cls):
        methods = set(method for interfaced_methods in cls.methods_map.values() for method in interfaced_methods)
        return methods

    @classmethod
    def is_known_method(cls, method=None):
        methodcasefold = [mm.casefold() for mm in cls.known_methods()]
        if method.casefold() in methodcasefold: return True
        else: return False
        
    def dump(self, filename=None, format='json'):
        model_dict = {'type': 'method'}
        for key in self.__dict__:
            tt = type(self.__dict__[key])
            if tt in [str, dict]:
                model_dict[key] = self.__dict__[key]
            model_dict['nthreads'] = self.nthreads

        if format == 'json':
            import json
            with open(filename, 'w') as fjson:
                json.dump(model_dict, fjson, indent=4)
        if format == 'dict':
            return model_dict

class meta_method(type):
    def __new__(cls, name, bases, namespace, available_methods=[]):
        new = super().__new__(cls, name, bases, namespace)
        if not available_methods:
            available_methods = methods.methods_map[name.split('_')[0]]
        new.available_methods = available_methods
        return new
    
    def is_available_method(self, method):
        return method.casefold() in [m.casefold() for m in self.available_methods]
    
        

def load(filename, format=None):
    '''
    Load a saved model object.
    '''
    if filename[-5:] == '.json' or format == 'json':
        try:
            return load_json(filename)
        except:
            pass
    if filename[-5:] == '.npz' or format == 'npz':
        try:
            return load_npz(filename)
        except:
            pass
    
    else:
        return load_pickle(filename)

def load_json(filename):
    import json
    with open(filename) as f:
        model_dict = json.load(f)
    return load_dict(model_dict)

def load_npz(filename):
    pass

def load_pickle(filename):
    import pickle
    with open(filename, 'rb') as file:
        return pickle.load(file)

def load_dict(model_dict):
    type = model_dict.pop('type')
    nthreads = model_dict.pop('nthreads') if 'nthreads' in model_dict else 0
    if type == 'method':
        kwargs = {}
        if 'kwargs' in model_dict:
            kwargs = model_dict.pop('kwargs')
        model = methods(**model_dict, **kwargs)

    if type == 'ml_model':
        model = globals()[model_dict['ml_model_type'].split('.')[-1]](**model_dict['kwargs'])

    if type == 'model_tree_node':
        children = [load_dict(child_dict) for child_dict in model_dict['children']] if model_dict['children'] else None
        name = model_dict['name']
        operator = model_dict['operator']
        model = load_dict(model_dict['model']) if model_dict['model'] else None
        weight = model_dict['weight'] if 'weight' in model_dict else None 
        model = model_tree_node(name=name, children=children, operator=operator, model=model)
        if weight:
            model.weight = weight
        
    model.set_num_threads(nthreads)
    return model

if __name__ == '__main__':
    pass