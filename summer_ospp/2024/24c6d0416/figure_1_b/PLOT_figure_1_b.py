import matplotlib.pyplot as plt
import numpy as np

layers = np.linspace(1,10,10)

adapt_un = [0.673573085,0.602932175,0.962652939,1.399264537,1.775244394,2.067374567,2.488940057,2.984423231,3.49024726,3.691840055]
adapt_ex = [0.532557164,0.755112105,0.950879667,1.316831743,1.615916095,1.706838032,1.973512957,2.448957764,2.699882632,2.869403462]
adapt_no = [0.0001768333,0.1787643027,0.2743962325,0.8736879091,1.3914055125,1.7474414126,1.9538014065,2.1176782566,2.2226361771,2.2662582386]
adapt_un2 = [0.0035032862,0.0824844933,0.3436524718,1.0215058329,1.5551618649,1.7554474912,1.9143476787,2.0160006873,2.0254657380,2.0658277480]

stand_un = [1.1041881410,1.8863991320,3.6122497660,5.1271808240,6.4936449320,7.3882893580,7.7462409270,8.2241848730,8.4792076930,8.5465426300]
stand_ex = [0.8677936040,2.2436096110,3.3057756880,4.8109415550,5.7872433390,6.6288151450,7.3164105930,7.6192882210,7.8893679420,8.0981927410]
stand_no = [0.6851887300,2.0006443610,3.3429008800,4.8445291080,6.1379538390,6.9349139430,7.5913193410,8.1230417540,8.3048330390,8.4459079620]
stand_un2 = [0.7058166841,1.9765877676,2.5431402669,4.7345758404,6.1197052397,7.2956232423,7.7296804141,8.2040680679,8.4897611692,8.5270383588]

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
