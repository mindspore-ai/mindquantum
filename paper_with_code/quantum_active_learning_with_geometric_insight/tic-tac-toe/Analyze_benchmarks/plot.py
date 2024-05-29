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


darkred='#8B0000'
darkblue='#00008B'


# with open('EQNN-Z_USAMP.pickle', 'rb') as f:
#     eqnnz_usamp = pickle.load(f)

with open('EQNN-Z_oracle.pickle', 'rb') as f:
    eqnnz_oracle = pickle.load(f)
    
with open('EQNN-Z_entropy.pickle', 'rb') as f:
    eqnnz_entropy = pickle.load(f)
with open('EQNN-Z_RS.pickle', 'rb') as f:
    eqnnz_rs = pickle.load(f)
# with open('BEL-Z_USAMP.pickle', 'rb') as f:
#     belz_usamp = pickle.load(f)
# with open('BEL-Z_RS.pickle', 'rb') as f:
#     belz_rs = pickle.load(f)
    
# with open('EQNN-Z_F.pickle', 'rb') as f:
#     eqnnz_F = pickle.load(f)

N_list = np.linspace(1,20,20,dtype=int)
N_oracle = np.linspace(3,20,18)
fig,ax=plt.subplots(1, 1, figsize=(8,4.5), sharey=False, sharex=False)





# plt.errorbar(N_list,np.mean(belz_rs,axis=0),np.std(belz_rs,axis=0),color='k',fmt='d',elinewidth=1,capsize=2,ms=3,label='BEL-Z-RS',alpha=0.9)
# plt.errorbar(N_list,np.mean(belz_usamp,axis=0),np.std(belz_usamp,axis=0),color='purple',fmt='s',elinewidth=1,capsize=2,ms=3,label='BEL-Z-USAMP',alpha=0.9)
# plt.errorbar(N_list,np.mean(eqnnz_F,axis=0),np.std(eqnnz_F,axis=0),color='g',fmt='x',elinewidth=1,capsize=2,ms=3,label='EQNN-Z-Fidelity',alpha=0.9)
plt.errorbar(N_list,np.mean(eqnnz_rs,axis=0),np.std(eqnnz_rs,axis=0),color=darkred,fmt='^',elinewidth=2,capsize=4,ms=4,label='Random sampling',alpha=0.9)
plt.errorbar(N_list,np.mean(eqnnz_entropy,axis=0),np.std(eqnnz_entropy,axis=0),color=darkblue,fmt='o',elinewidth=2,capsize=4,ms=4,label='Entropy sampling',alpha=0.9)
plt.errorbar(N_oracle,np.mean(eqnnz_oracle,axis=0),np.std(eqnnz_oracle,axis=0),color='k',fmt='d',elinewidth=2,capsize=4,ms=4,label='Oracle',alpha=0.9)



plt.xticks([1,5,10,15,20],fontsize=30)
plt.yticks([0.5,0.6,0.7],fontsize=30)
plt.xlabel('Number of labels',fontsize=30)
plt.ylabel('Correct rate',fontsize=30)
plt.legend(frameon = False,fontsize=10)
fig.savefig('entropy.pdf',dpi=800,bbox_inches='tight',format='pdf')





