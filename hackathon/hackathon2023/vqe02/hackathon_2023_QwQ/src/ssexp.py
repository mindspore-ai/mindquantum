#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/02 

# implementation of subspace-expansion in "https://arxiv.org/abs/1603.05681"

from .common import *
from .vqe import run as run_gs


def run_ssexp(mol:MolecularData, ham:Ham, config:Config) -> Tuple[QVM, float, Params]:
  pass


def ssexp_solver(mol:MolecularData, config:Config) -> float:
  ham = get_ham(mol, config)

  # Ground state E0: |ψ(λ0)>
  gs_sim, gs_ene, params = run_gs(mol, ham, config)

  # Excited state E1: |ψ(λ1)>
  es_ene = run_ssexp()

  return es_ene
