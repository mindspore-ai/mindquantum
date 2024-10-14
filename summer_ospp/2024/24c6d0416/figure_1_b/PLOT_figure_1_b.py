import matplotlib.pyplot as plt
import numpy as np

layers = np.linspace(1,10,10)

adapt_un = [0.0007290606,	0.0268325167,	0.0197428540,	0.6057778878,	1.2139263222,	1.2986768641,	1.4242736971,	1.4432156265,	1.4526428011,	1.4643632345]
adapt_ex = [0.0000512419,	0.0000463667,	0.0004084033,	0.1357294244,	0.8251305271,	0.8518935589,	0.8547432329,	0.8612539061,	0.8700068024,	0.8814140178]
adapt_no = [0.0000000000,	0.0000000000,	0.0000000000,	0.0000000004,	0.6171003316,	0.6177105791,	0.6197784622,	0.6233035984,	0.6282828545,	0.6347098328]
adapt_un2 = [0.0000000001,	0.0000000001,	0.0000000002,	0.0765816970,	0.6730580302,	0.6811545615,	0.6890380265,	0.6914665833,	0.6943635981,	0.6977270442]

stand_un = [0.5624574482,	1.3567059841,	2.3384085953,	3.7410349986,	4.8568181838,	5.5400062265,	5.9601019959,	6.1382534678,	6.1687906874,	6.3484101338]
stand_ex = [0.4209612853,	1.1475994333,	1.7376789048,	3.0001925284,	3.9601642049,	4.6674243296,	5.1851734468,	5.3788221975,	5.3095518984,	5.8196793651]
stand_no = [0.2905980402,	1.2191653566,	1.7940955535,	3.4430514757,	4.5894970527,	5.3947798575,	5.8497133633,	6.0632598439,	6.0120193197,	6.2659054552]
stand_un2 = [0.2965201327,	1.2452756072,	1.9029175226,	3.7630862631,	4.8409975577,	5.7138847042,	6.0792604265,	6.2478844309,	6.2652283982,	6.3708779040]

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.plot(layers, stand_un, color='purple', linestyle='-', marker='^', markersize=7, label='QAOA U[0,1]')
plt.plot(layers, stand_ex, color='purple', linestyle=':', marker='8', markersize=7, label='QAOA Exp(1)')
plt.plot(layers, stand_no, color='purple', linestyle='-',marker='h', markersize=7, label='QAOA N[0,1]')
plt.plot(layers, stand_un2, color='purple', linestyle='--', marker='*', markersize=7, label='QAOA U[-1,1]')

plt.plot(layers, adapt_un, color='green', linestyle='-', marker='d', markersize=7)
plt.plot(layers, adapt_ex, color='green', linestyle=':', marker='o', markersize=7)
plt.plot(layers, adapt_no, color='green', linestyle='-',marker='s', markersize=7)
plt.plot(layers, adapt_un2, color='green', linestyle='--', marker='x', markersize=7)
plt.plot(layers,np.full_like(layers,0),color='grey',linestyle='--',linewidth=3)

plt.xlabel('Numbers of layers p', fontsize=16)
plt.ylabel('E[S]', fontsize=16)
plt.tight_layout()
plt.legend(fontsize=14)
plt.show()
