import numpy as np
import matplotlib.pyplot as plt

qubits = [10,34,58,83,107,150,200]
value = [-0.30990902,-0.352549399,-0.571241367,-0.646966646,-0.799231086,-0.82862472,-0.91423538,
    -0.993094679,-1.035155966,-1.126789509,-1.229782963,-1.302425743,-1.24101199,-1.465110786]

error = [0.094835291,0.034506409,0.090915626,0.031996991,0.071613785,0.044328358,0.07458878,
    0.046479193,0.064498372,0.041883992,0.046341297,0.057208325,0.083661025,0.014922476]

gw_value = [value[i] for i in range(14) if i % 2 == 0]
adapt_value = [value[i] for i in range(14) if i % 2 == 1]

gw_yerr = [error[i] for i in range(14) if i % 2 == 0]
adapt_yerr = [error[i] for i in range(14) if i % 2 == 1]

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.errorbar(qubits, gw_value, yerr=gw_yerr, fmt='o', linestyle='--',ecolor='gray', capsize=5, capthick=2,color = 'gray',label='GW')
plt.errorbar(qubits, adapt_value, yerr=adapt_yerr, fmt='-o', ecolor='green', capsize=5, capthick=2,color = 'green',label='ADAPT Clifford ri')
plt.xlabel('Number of qubits', fontsize=16)
plt.ylabel('E[Emin]/N', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

diff = [(adapt_value[i] - gw_value[i]) * qubits[i] for i in range(7)]

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.plot(qubits, diff, color='gray',marker='o',markersize=5)
plt.plot(qubits, np.full_like(qubits,0), color='red',linestyle=':')
plt.xlabel('Number of qubits', fontsize=16)
plt.ylabel('E[Ecliff - Egw]', fontsize=16)
plt.tight_layout()
plt.show()

