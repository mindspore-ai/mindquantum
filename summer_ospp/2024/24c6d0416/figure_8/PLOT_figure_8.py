import matplotlib.pyplot as plt
import numpy as np

node = 50
edge_P = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gw_0 = [-33.75, -46.85, -50.45, -55.4, -55.1, -55.45, -55.3, -51.1, -34.55]
gw_1 = [-37.45, -50.95, -59.35, -62.1, -63.8, -55.45, -62.6, -55.2, -45.35]
gw_2 = [-37.75, -51.85, -59.65, -64.1, -66.2, -65.15, -65.4, -57.9, -46.55]
gw_3 = [-37.75, -51.85, -60.15, -64.6, -66.8, -65.75, -65.7, -58.2, -46.75]
gw_4 = [-37.95, -51.85, -60.35, -64.7, -66.9, -65.85, -65.9, -58.4, -46.95]
rand = [-35.15, -46.35, -53.75, -59.8, -61.5, -60.65, -61.3, -54.9, -44.25]
deter = [-37.75, -51.85, -60.05, -64.5, -66.7, -65.05, -65.6, -58, -46.95]

gw_0_yerr = [0.039306488,0.100104945,0.085023526,0.132499057,0.171976743,0.114494541,0.133880544,0.046861498,0.192013020]
gw_1_yerr = [0.035902646,0.052430907,0.075239617,0.072498276,0.055892754,0.052048055,0.076000000,0.043174066,0.067977938]
gw_2_yerr = [0.032634338,0.048795492,0.073898579,0.050358713,0.052191953,0.028653098,0.056709788,0.032802439,0.063474404]
gw_3_yerr = [0.032634338,0.048795492,0.063725976,0.050950957,0.047791213,0.026172505,0.056603887,0.028705400,0.061359596]
gw_4_yerr = [0.033600595,0.048795492,0.060174746,0.047159304,0.049959984,0.025709920,0.057930993,0.028565714,0.059236813]
rand_yerr = [0.039509493,0.057105166,0.072972598,0.057654141,0.078230429,0.055686623,0.086740994,0.045563143,0.095105205]
deter_yerr = [0.037215588,0.048795492,0.065946948,0.049396356,0.046303348,0.025709920,0.049152823,0.031622777,0.056471232]

colors = [f'#{i:02x}{i:02x}{i:02x}' for i in np.linspace(0, 255, 6).astype(int)]

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.errorbar(edge_P, [x/50 for x in gw_0], yerr=gw_0_yerr, fmt='x', linestyle='--',ecolor=colors[4], capsize=5, capthick=2,color =colors[4],label='GW')
plt.errorbar(edge_P, [x/50 for x in rand], yerr=rand_yerr, fmt='x', linestyle='-',ecolor='green', capsize=5, capthick=2,color ='green',label='ADAPT Clifford ri')
plt.xticks([0.25,0.50,0.75])
plt.xlabel('Edge probability', fontsize=16)
plt.ylabel('E[Emin]/N', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

diff = [rand[i] - gw_0[i] for i in range(9)]

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.plot(edge_P, diff, color='gray',marker='x',markersize=5)
plt.plot(edge_P, np.full_like(edge_P,0), color='red',linestyle=':')
plt.xticks([0.25,0.50,0.75])
plt.ylim(-11,0.8)
plt.xlabel('Edge probability', fontsize=16)
plt.ylabel('E[Ecliff - Egw]', fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.xticks([0.25,0.50,0.75])
plt.errorbar(edge_P, [x/50 for x in deter], yerr=deter_yerr, fmt='x', linestyle='-',ecolor='#E65100', capsize=5, capthick=2,color ='#E65100',label='ADAPT Clifford')
plt.errorbar(edge_P, [x/50 for x in gw_4], yerr=gw_4_yerr, fmt='x', linestyle='--',ecolor='black', capsize=5, capthick=2,color ='black',label='GW with I = 10^4')
plt.errorbar(edge_P, [x/50 for x in gw_0], yerr=gw_4_yerr, fmt='x', linestyle='--',ecolor=colors[4], capsize=5, capthick=2,color =colors[4],label='GW with I = 10^0')
plt.xlabel('Edge probability', fontsize=16)
plt.ylabel('E[Emin]/N', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

gw0_diff = [deter[i] - gw_0[i] for i in range(9)]
gw1_diff = [deter[i] - gw_1[i] for i in range(9)]
gw2_diff = [deter[i] - gw_2[i] for i in range(9)]
gw3_diff = [deter[i] - gw_3[i] for i in range(9)]
gw4_diff = [deter[i] - gw_4[i] for i in range(9)]

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.xticks([0.25,0.50,0.75])
plt.plot(edge_P,np.full_like(edge_P,0),color='red',linestyle=':',linewidth=3)
plt.plot(edge_P, gw0_diff, color=colors[4], linestyle='-',marker='x', markersize=7,label='I = 10^0')
plt.plot(edge_P, gw1_diff, color=colors[3], linestyle='-', marker='x', markersize=7,label='I = 10^1')
plt.plot(edge_P, gw2_diff, color=colors[2], linestyle='-', marker='x', markersize=7,label='I = 10^2')
plt.plot(edge_P, gw3_diff, color=colors[1], linestyle='-', marker='x', markersize=7,label='I = 10^3')
plt.plot(edge_P, gw4_diff, color=colors[0], linestyle='-', marker='x', markersize=7,label='I = 10^4')
plt.xlabel('Edge probability', fontsize=16)
plt.ylabel('E[Ecliff - Egw]', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()