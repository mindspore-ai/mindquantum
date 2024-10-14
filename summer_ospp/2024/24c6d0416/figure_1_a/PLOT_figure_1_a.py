import matplotlib.pyplot as plt
import numpy as np

layer_f1 = np.linspace(1, 10, 10)

qaoa_u = [0.477651871, 0.640366004, 0.755479353,	0.833281112,	0.878973432,	0.909416851,	0.929633459,	0.942386022,	0.951095962,	0.95921513]
qaoa_e = [0.515773993, 0.695324761, 0.795487918,	0.857380491,	0.896369904,	0.923106128,	0.943198739,	0.956521742,	0.965889195,	0.972255497]
qaoa_n = [0.555428511, 0.744562557,	0.855147716,	0.909228157,	0.935970004,	0.954785733,	0.967462415,	0.975754874,	0.982716846,	0.98690489]
qaoa_u2 = [0.533105674, 0.736863055,	0.852494126,	0.908443911,	0.935388945,	0.952900343,	0.966081926,	0.974428984,	0.98073626,	0.984586997]

adapt_u = [0.285999852,	0.535369625,	0.723792252,	0.865741634,	0.946705329,	0.961513569,	0.967612421,	0.969515894,	0.969515894,	0.969515894]
adapt_e = [0.38420097,	0.626915094,	0.77907286,	0.884133346,	0.950002235,	0.96197749,	0.963038966,	0.963102511,	0.96310466,	0.96310891]
adapt_n = [0.267535908,	0.512121863,	0.719669874,	0.866140913,	0.958070096,	0.959254677,	0.95927831,	0.959337258,	0.959400961,	0.961227737]
adapt_u2 = [0.207465273,	0.441431985,	0.67389765,	0.853790805,	0.949392071,	0.953851972,	0.954130971,	0.954220928,	0.95704279,	0.95713634]

adapt_u_opt = [0.285938847,	0.534193521,	0.725509088,	0.873884357,	0.965888556,	0.982716102,	0.989646615,	0.991809653,	0.991809653,	0.991809653]
adapt_e_opt = [0.382927267,	0.630644903,	0.788611442,	0.898562419,	0.976977764,	0.991234019,	0.992497681,	0.99257333,	0.992575888,	0.992580948]
adapt_n_opt = [0.264466647,	0.51238088,	0.726163753,	0.883854545,	0.995477835,	0.996922446,	0.996951267,	0.997023155,	0.997100841,	0.999328618]
adapt_u2_opt = [0.200260911,	0.437234288,	0.684694464,	0.875212327,	0.986688398,	0.992260011,	0.99260876,	0.992721206,	0.996248532,	0.996365471]

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

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.plot(layer_f1, qaoa_u, color='purple', linestyle='-', marker='^', markersize=7)
plt.plot(layer_f1, qaoa_e, color='purple', linestyle=':', marker='8', markersize=7)
plt.plot(layer_f1, qaoa_n, color='purple', linestyle='-',marker='h', markersize=7)
plt.plot(layer_f1, qaoa_u2, color='purple', linestyle='--', marker='*', markersize=7)

plt.plot(layer_f1, adapt_u_opt, color='green', linestyle='-', marker='d', markersize=7, label='ADAPT-QAOA U[0,1]')
plt.plot(layer_f1, adapt_e_opt, color='green', linestyle=':', marker='o', markersize=7, label='ADAPT-QAOA Exp(1)')
plt.plot(layer_f1, adapt_n_opt, color='green', linestyle='-',marker='s', markersize=7, label='ADAPT-QAOA N(0,1)')
plt.plot(layer_f1, adapt_u2_opt, color='green', linestyle='--', marker='x', markersize=7, label='ADAPT-QAOA U[-1,1]')
plt.plot(layer_f1,np.full_like(layer_f1,1),color='grey',linestyle='--',linewidth=3)

plt.xlabel('Numbers of layers p', fontsize=16)
plt.ylabel('Approximation ratio', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()


