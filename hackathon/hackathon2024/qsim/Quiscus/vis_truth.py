#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/01

# 查看各种 H4 分子几何构型的真实基态能量

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils import generate_molecule

BASE_PATH = Path(__file__).parent
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)
FCI_FILE = IMG_PATH / 'fci.json'

get_fci = lambda geometry: generate_molecule(geometry).fci_energy


def get_fci_linear(dist:float=1.0) -> float:
  ''' 直线 '''
  geometry = [
    ['H', [0.0, 0.0, 0.0 * dist]],
    ['H', [0.0, 0.0, 1.0 * dist]],
    ['H', [0.0, 0.0, 2.0 * dist]],
    ['H', [0.0, 0.0, 3.0 * dist]],
  ]
  return get_fci(geometry)

def get_fci_square(dist:float=1.0) -> float:
  ''' 正方形 '''
  geometry = [
    ['H', [0.0, 0.0, 0.0]],
    ['H', [0.0, 0.0, dist]],
    ['H', [0.0, dist, 0.0]],
    ['H', [0.0, dist, dist]],
  ]
  return get_fci(geometry)

def get_fci_diamond(dist:float=1.0) -> float:
  ''' 60°菱形 '''
  geometry = [
    ['H', [0.0, 0.0, +np.cos(np.pi/3) * dist]],
    ['H', [0.0, 0.0, -np.cos(np.pi/3) * dist]],
    ['H', [0.0, +np.sin(np.pi/3) * dist, 0.0]],
    ['H', [0.0, -np.sin(np.pi/3) * dist, 0.0]],
  ]
  return get_fci(geometry)

def get_fci_Y(dist:float=1.0) -> float:
  ''' 正Y字 '''
  geometry = [
    ['H', [0.0, 0.0, 0.0]],
    ['H', [0.0, 0.0, dist]],
    ['H', [0.0, +np.sin(np.pi/3) * dist, -np.cos(np.pi/3) * dist]],
    ['H', [0.0, -np.sin(np.pi/3) * dist, -np.cos(np.pi/3) * dist]],
  ]
  return get_fci(geometry)

def get_fci_diagonal(dist:float=1.0) -> float:
  ''' 正交轴 '''
  geometry = [
    ['H', [0.0, 0.0, 0.0]],
    ['H', [dist, 0.0, 0.0]],
    ['H', [0.0, dist, 0.0]],
    ['H', [0.0, 0.0, dist]],
  ]
  return get_fci(geometry)

def get_fci_tetrahedron(dist:float=1.0) -> float:
  ''' 正四面体 '''
  geometry = [
    ['H', [0.0, 0.0, 0.0]],
    ['H', [dist, 0.0, 0.0]],
    ['H', [dist/2, np.sqrt(3)*dist/2, 0.0]],
    ['H', [dist/2, np.sqrt(3)*dist/6, np.sqrt(6)*dist/3]],
  ]
  return get_fci(geometry)


def load_db():
  try:
    assert FCI_FILE.exists()
    with open(FCI_FILE, 'r', encoding='utf-8') as fh:
      fci_db = json.load(fh)
  except:
      fci_db = {}
  return fci_db

def save_db(fci_db):
  print(f'>> save to {FCI_FILE}')
  with open(FCI_FILE, 'w', encoding='utf-8') as fh:
    json.dump(fci_db, fh, indent=2, ensure_ascii=False)


MORPHS = {name[len('get_fci_'):]: value for name, value in globals().items() if name.startswith('get_fci_')}

fci_db = load_db()
try:
  for name, get_fci_fn in MORPHS.items():
    fp = IMG_PATH / f'H4-fci_{name}.png'
    if fp.exists(): continue
    fci_db[name] = {}

    xs, ys = [], []
    for i in range(30, 250+1):
      dist = i / 100   # [0.30, 2.50]
      fci = get_fci_fn(dist)
      fci_db[name][dist] = fci
      xs.append(dist)
      ys.append(fci)

    plt.plot(xs, ys)
    plt.suptitle(f'H4-fci: {name}')
    plt.savefig(fp, dpi=600)
    plt.close()
    print(f'>> savefig to {fp}')
except KeyboardInterrupt:
  print('>> Exit by Ctrl+C')
finally:
  save_db(fci_db)
