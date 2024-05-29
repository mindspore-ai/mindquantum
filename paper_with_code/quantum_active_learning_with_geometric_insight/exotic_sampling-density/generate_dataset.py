#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:06:58 2024
without any symmetry, just an example
@author: jonzen
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import random

from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'sans'
mpl.rcParams['mathtext.it'] = 'sans:italic'
mpl.rcParams['mathtext.default'] = 'it'

np.random.seed(16)

data = []


for i in range(4):
    x = 0.1 + 0.05 * np.random.random()
    y = 0.05 + 0.05 * np.random.random()
    label = 1
    temp = [x,y],label
    data.append(temp)

for i in range(4):
    x = 0.1 + 0.05 * np.random.random()
    y = -0.05 + 0.05 * np.random.random()
    label = -1
    temp = [x,y],label
    data.append(temp)
temp = [-0.1,0.01],1
data.append(temp)

with open('training_set.pickle', 'wb') as f:
    pickle.dump(data, f)