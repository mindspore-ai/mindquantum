import matplotlib.pyplot as plt
import numpy as np

layer_f1 = np.linspace(1, 10, 10)

qaoa_u = [0.490960589, 0.630944168, 0.773625115, 0.828101512, 0.85901052, 0.887354128, 0.909105972, 0.921146402, 0.930528257, 0.941470161]
qaoa_e = [0.472112144, 0.568576995, 0.643016596, 0.659209353, 0.712570823, 0.741539409, 0.753018981, 0.784351935, 0.789560239, 0.85014111]
qaoa_n = [0.507115863, 0.641781487, 0.799686366, 0.853150108, 0.883513094, 0.915942239, 0.938350953, 0.948998854, 0.952983249, 0.961587728]
qaoa_u2 = [0.567017548, 0.67146036, 0.846826981, 0.888336462, 0.898274752, 0.908626983, 0.918596609, 0.926739045, 0.936151769, 0.950552179]

adapt_u = [0.309063419, 0.341465432, 0.424540391, 0.564786684, 0.629163118, 0.772578754, 0.869137505, 0.906532399, 0.93891835, 0.986818307]
adapt_e = [0.331054022, 0.380707029, 0.473600654, 0.525951541, 0.606557809, 0.730217174, 0.817106694, 0.943031161, 0.996936895, 0.998382458]
adapt_n = [0.247198807, 0.434953003, 0.567347225, 0.794796675, 0.950441794, 0.985968592, 0.992197152, 0.992197152, 0.992197152, 0.992197152]
adapt_u2 = [0.174620965, 0.322572804, 0.479907259, 0.693754341, 0.86370607, 0.951614625, 0.990202218, 0.99971987, 0.999999999, 1]

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.plot(layer_f1, qaoa_u, color='purple', linestyle='-', marker='^', markersize=7)
plt.plot(layer_f1, qaoa_e, color='purple', linestyle=':', marker='8', markersize=7)
plt.plot(layer_f1, qaoa_n, color='purple', linestyle='-',marker='h', markersize=7)
plt.plot(layer_f1, qaoa_u2, color='purple', linestyle='--', marker='*', markersize=7)

plt.plot(layer_f1, adapt_u, color='green', linestyle='-', marker='d', markersize=7, label='ADAPT-QAOA U[0,1]')
plt.plot(layer_f1, adapt_e, color='green', linestyle=':', marker='o', markersize=7, label='ADAPT-QAOA Exp(1)')
plt.plot(layer_f1, adapt_n, color='green', linestyle='-',marker='s', markersize=7, label='ADAPT-QAOA N(0,1)')
plt.plot(layer_f1, adapt_u2, color='green', linestyle='--', marker='x', markersize=7, label='ADAPT-QAOA U[-1,1]')
plt.plot(layer_f1,np.full_like(layer_f1,1),color='grey',linestyle='--',linewidth=3)

plt.xlabel('Numbers of layers p', fontsize=16)
plt.ylabel('Approximation ratio', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()



