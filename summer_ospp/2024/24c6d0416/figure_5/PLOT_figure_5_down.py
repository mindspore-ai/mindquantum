import matplotlib.pyplot as plt
import numpy as np

qubits = [10,34,58,82,106,150,200]
qubits_un = [10,34,58,82,106,150,200,250]

GW_un = [-5.35, -18.65, -32.7, -45.9, -61.2, -85.25, -113.2, -150.35]
ran_un = [-5.35, -19.2, -33.6, -46.65, -60.25, -85.5, -115.4, -145]

GW_we = [-2.749806242, -9.861414649, -17.08443199, -25.64227543, -33.22931228, -46.58260898, -62.53239129]
ran_we = [-3.019046497, -10.65979757, -18.34526988, -27.06912246, -34.72695228, -49.44244701, -65.6307904]

GW_re = [-3.605578385, -15.74677969, -28.59042226, -41.36170725, -55.15087687, -78.69663609, -102.8031623]
ran_re = [-3.707750679, -17.83388086, -31.12771561, -43.90549428, -57.4866164, -81.55359502, -108.5986633]

GW_er = [-5.625, -32.6, -74.4, -124.375, -183.85, -300.225, -457.2]
ran_er = [-5.725, -35.25, -77.4, -131.725, -191.4, -323.125, -493]


diff_un = [ran_un[i]-GW_un[i] for i in range(8)]
diff_we = [ran_we[i]-GW_we[i] for i in range(7)]
diff_re = [ran_re[i]-GW_re[i] for i in range(7)]
diff_er = [ran_er[i]-GW_er[i] for i in range(7)]

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.plot(qubits_un, diff_un, color='gray', linestyle='-', marker='o', markersize=7)
plt.plot(qubits_un,np.full_like(qubits_un,0),color='red',linestyle=':',linewidth=3)
plt.xlabel('Numbers of nodes', fontsize=16)
plt.ylabel('E[Eclifford - Egw]', fontsize=16)
plt.title('unweighted 3-regular', fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.plot(qubits, diff_we, color='gray', linestyle='-', marker='D', markersize=7)
plt.plot(qubits,np.full_like(qubits,0),color='red',linestyle=':',linewidth=3)
plt.xlabel('Numbers of nodes', fontsize=16)
plt.ylabel('E[Eclifford - Egw]', fontsize=16)
plt.title('weighted 3-regular', fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.plot(qubits, diff_re, color='gray', linestyle='-', marker='s', markersize=7)
plt.plot(qubits,np.full_like(qubits,0),color='red',linestyle=':',linewidth=3)
plt.xlabel('Numbers of nodes', fontsize=16)
plt.ylabel('E[Eclifford - Egw]', fontsize=16)
plt.title('weighted 8-regular', fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.plot(qubits, diff_er, color='gray', linestyle='-', marker='x', markersize=7)
plt.plot(qubits,np.full_like(qubits,0),color='red',linestyle=':',linewidth=3)
plt.xlabel('Numbers of nodes', fontsize=16)
plt.ylabel('E[Eclifford - Egw]', fontsize=16)
plt.title('Erdos Yenyi', fontsize=16)
plt.tight_layout()
plt.show()
