# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:53:14 2024
Generate dataset with Z2 symmetry in data label
@author: jonze
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

N = 500
r_list = np.random.normal(0.5, 0.15, N)
print('max,min=',r_list.max(),r_list.min(),'0<r<1 verification')
theta_list = np.random.uniform(0,2*np.pi,N)

data = []
x0_list = []
x1_list = []
y_list = []
for i in range(N):
    value = np.cos(2*theta_list[i]+0.58) #periodic
    if value >= 0:
        ylabel = 1
    else:
        ylabel = -1
    x0_list.append(r_list[i]*np.cos(theta_list[i]))
    x1_list.append(r_list[i]*np.sin(theta_list[i]))
    y_list.append(ylabel)
    temp = [x0_list[i],x1_list[i]],ylabel
    data.append(temp)
    
#visualize
# cm_pt = matplotlib.colors.ListedColormap(["blue", "red"])
# fig,ax=plt.subplots(1, 1, figsize=(4,4), sharey=False, sharex=False)
# plt.scatter(x0_list[0:300],x1_list[0:300],c=y_list[0:300],cmap=cm_pt,marker='.')
# plt.title('Training set')
# plt.xlim([-1,1])
# plt.ylim([-1,1])
# plt.xticks([-1,-0.5,0,0.5,1])
# plt.yticks([-1,-0.5,0,0.5,1])
# fig.savefig('training_set.pdf',dpi=800,bbox_inches='tight',format='pdf')

# cm_pt = matplotlib.colors.ListedColormap(["blue", "red"])
# fig,ax=plt.subplots(1, 1, figsize=(4,4), sharey=False, sharex=False)
# plt.scatter(x0_list[300:400],x1_list[300:400],c=y_list[300:400],cmap=cm_pt,marker='.')
# plt.title('Verification set')
# plt.xlim([-1,1])
# plt.ylim([-1,1])
# plt.xticks([-1,-0.5,0,0.5,1])
# plt.yticks([-1,-0.5,0,0.5,1])
# fig.savefig('val_set.pdf',dpi=800,bbox_inches='tight',format='pdf')

# cm_pt = matplotlib.colors.ListedColormap(["blue", "red"])
# fig,ax=plt.subplots(1, 1, figsize=(4,4), sharey=False, sharex=False)
# plt.scatter(x0_list[400:500],x1_list[400:500],c=y_list[400:500],cmap=cm_pt,marker='.')
# plt.title('Testing set')
# plt.xlim([-1,1])
# plt.ylim([-1,1])
# plt.xticks([-1,-0.5,0,0.5,1])
# plt.yticks([-1,-0.5,0,0.5,1])
# fig.savefig('test_set.pdf',dpi=800,bbox_inches='tight',format='pdf')


darkred='#8B0000'
darkblue='#00008B'
cm_pt = matplotlib.colors.ListedColormap([darkblue, darkred])
fig,ax=plt.subplots(1, 1, figsize=(4,4), sharey=False, sharex=False)
#plt.scatter(x0_list,x1_list,c=y_list,cmap=cm_pt,marker='.')

for i in range(len(x0_list)):
    if y_list[i] == -1:
        plt.scatter(x0_list[i],x1_list[i],c=darkblue,marker='x')
    else:
        plt.scatter(x0_list[i],x1_list[i],c=darkred,marker='.')
#trick
plt.scatter(100,100,c=darkblue,marker='x',label='Class 0')
plt.scatter(100,100,c=darkred,marker='.',label='Class 1')
plt.title('Data set',fontsize=15)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.legend(fontsize=10,ncol=1,frameon=False)
plt.xticks([-1,-0.5,0,0.5,1],fontsize=15)
plt.yticks([-1,-0.5,0,0.5,1],fontsize=15)
plt.xlabel('$x_0$',fontsize=15)
plt.ylabel('$x_1$',fontsize=15)
fig.savefig('dataset.pdf',dpi=800,bbox_inches='tight',format='pdf')


training_set = data[0:300]
val_set = data[300:400]
test_set = data[400:500]

# with open('training_set.pickle', 'wb') as f:
#     pickle.dump(training_set, f)
# with open('val_set.pickle', 'wb') as f:
#     pickle.dump(val_set, f)
# with open('test_set.pickle', 'wb') as f:
#     pickle.dump(test_set, f)




