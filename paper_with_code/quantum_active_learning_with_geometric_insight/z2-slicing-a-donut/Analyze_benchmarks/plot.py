# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:40:46 2024
Compare
@author: jonze
"""
import math
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib


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

with open('EQNN-Z_USAMP.pickle', 'rb') as f:
    eqnnz_usamp = pickle.load(f)
with open('EQNN-Z_RS.pickle', 'rb') as f:
    eqnnz_rs = pickle.load(f)
with open('BEL-Z_USAMP.pickle', 'rb') as f:
    belz_usamp = pickle.load(f)
with open('BEL-Z_RS.pickle', 'rb') as f:
    belz_rs = pickle.load(f)
    
with open('EQNN-Z_F.pickle', 'rb') as f:
    eqnnz_F = pickle.load(f)

N_list = np.linspace(1,20,20,dtype=int)
fig,ax=plt.subplots(1, 1, figsize=(8,4.5), sharey=False, sharex=False)


darkred='#8B0000'
darkblue='#00008B'
darkgreen='#008B00'


plt.errorbar(N_list,np.mean(belz_rs,axis=0),np.std(belz_rs,axis=0),color='k',fmt='d',elinewidth=2,capsize=4,ms=4,label='HEA-Z-RS',alpha=0.9)
plt.errorbar(N_list,np.mean(belz_usamp,axis=0),np.std(belz_usamp,axis=0),color='purple',fmt='s',elinewidth=2,capsize=4,ms=4,label='HEA-Z-USAMP',alpha=0.9)
plt.errorbar(N_list,np.mean(eqnnz_F,axis=0),np.std(eqnnz_F,axis=0),color=darkgreen,fmt='x',elinewidth=2,capsize=4,ms=4,label='EQNN-Z-FSAMP',alpha=0.9)
plt.errorbar(N_list,np.mean(eqnnz_rs,axis=0),np.std(eqnnz_rs,axis=0),color=darkred,fmt='^',elinewidth=2,capsize=4,ms=4,label='EQNN-Z-RS',alpha=0.9)
plt.errorbar(N_list,np.mean(eqnnz_usamp,axis=0),np.std(eqnnz_usamp,axis=0),color=darkblue,fmt='o',elinewidth=2,capsize=4,ms=4,label='EQNN-Z-USAMP',alpha=0.9)

plt.xticks([1,5,10,15,20],fontsize=30)
plt.yticks([0.5,0.6,0.7,0.8,0.9],fontsize=30)
plt.xlabel('Number of labels',fontsize=30)
plt.ylabel('Correct rate',fontsize=30)
plt.legend(frameon = False,fontsize=10)
fig.savefig('benchmark.pdf',dpi=800,bbox_inches='tight',format='pdf')





