import matplotlib.pyplot as plt

qubits = [10,34,58,82,106,150,200]
qubits_un = [10,34,58,82,106,150,200,250]
GW_un = [-5.35/10, -18.65/34, -32.7/58, -45.9/82, -61.2/106, -85.25/150, -113.2/200, -150.35/250]
GW_un_yerr = [0.101365675,0.046759156,0.040654573,0.030706532,0.023127643,0.019986106,0.021714051, 0.009981984]
ran_un = [-5.35/10, -19.2/34, -33.6/58, -46.65/82, -60.25/106, -85.5/150, -115.4/200, -145/250]
ran_un_yerr = [0.096306801,0.030987217,0.03127303,0.023544798,0.02308431,0.020763215,0.018466185, 0.011419282]

GW_we = [-2.749806242/10, -9.861414649/34, -17.08443199/58, -25.64227543/82, -33.22931228/106, -46.58260898/150, -62.53239129/200]
GW_we_yerr = [0.069302224,0.027289836,0.029089463,0.022111741,0.021390926,0.015849441,0.008310817]
ran_we = [-3.019046497/10, -10.65979757/34, -18.34526988/58, -27.06912246/82, -34.72695228/106, -49.44244701/150, -65.6307904/200]
ran_we_yerr = [0.052273698,0.017632968,0.02240681,0.014329193,0.013886824,0.013892036,0.009949042]

GW_re = [-3.605578385/10, -15.74677969/34, -28.59042226/58, -41.36170725/82, -55.15087687/106, -78.69663609/150, -102.8031623/200]
GW_re_yerr = [0.05905587,0.067744018,0.061377346,0.041212427,0.032886809,0.031826922,0.02611407]
ran_re = [-3.707750679/10, -17.83388086/34, -31.12771561/58, -43.90549428/82, -57.4866164/106, -81.55359502/150, -108.5986633/200]
ran_re_yerr = [0.051540991,0.03085217,0.02962091,0.023384945,0.022285867,0.01967911,0.015384383]

GW_er = [-5.625/10, -32.6/34, -74.4/58, -124.375/82, -183.85/106, -300.225/150, -457.2/200]
GW_er_yerr = [0.075622417,0.103004182,0.118031063,0.101938291,0.101103074,0.087803284,0.0917006]
ran_er = [-5.725/10, -35.25/34, -77.4/58, -131.725/82, -191.4/106, -323.125/150, -493/200]
ran_er_yerr = [0.0580409340,0.062823557,0.090685394,0.075852164,0.061981204,0.074161048,0.114825955]

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.errorbar(qubits_un, GW_un, yerr=GW_un_yerr, fmt='o', linestyle='--',ecolor='gray', capsize=5, capthick=2,color ='gray',label='GW')
plt.errorbar(qubits_un, ran_un, yerr=ran_un_yerr, fmt='o', linestyle='-',ecolor='green', capsize=5, capthick=2,color ='green',label='ADAPT Clifford ri')
plt.xticks(qubits_un)
plt.xlabel('Numbers of nodes', fontsize=16)
plt.ylabel('E[Emin]/N', fontsize=16)
plt.title('unweighted 3-regular', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.errorbar(qubits, GW_we, yerr=GW_we_yerr, fmt='D', linestyle='--',ecolor='gray', capsize=5, capthick=2,color ='gray',label='GW')
plt.errorbar(qubits, ran_we, yerr=ran_we_yerr, fmt='D', linestyle='-',ecolor='green', capsize=5, capthick=2,color ='green',label='ADAPT Clifford ri')
plt.xticks(qubits)
plt.xlabel('Numbers of nodes', fontsize=16)
plt.ylabel('E[Emin]/N', fontsize=16)
plt.title('weighted 3-regular', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.errorbar(qubits, GW_re, yerr=GW_re_yerr, fmt='s', linestyle='--',ecolor='gray', capsize=5, capthick=2,color ='gray',label='GW')
plt.errorbar(qubits, ran_re, yerr=ran_re_yerr, fmt='s', linestyle='-',ecolor='green', capsize=5, capthick=2,color ='green',label='ADAPT Clifford ri')
plt.xticks(qubits)
plt.xlabel('Numbers of nodes', fontsize=16)
plt.ylabel('E[Emin]/N', fontsize=16)
plt.title('weighted 8-regular', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.errorbar(qubits, GW_er, yerr=GW_er_yerr, fmt='x', linestyle='--',ecolor='gray', capsize=5, capthick=2,color ='gray',label='GW')
plt.errorbar(qubits, ran_er, yerr=ran_er_yerr, fmt='x', linestyle='-',ecolor='green', capsize=5, capthick=2,color ='green',label='ADAPT Clifford ri')
plt.xticks(qubits)
plt.xlabel('Numbers of nodes', fontsize=16)
plt.ylabel('E[Emin]/N', fontsize=16)
plt.title('Erdos Renyi', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

