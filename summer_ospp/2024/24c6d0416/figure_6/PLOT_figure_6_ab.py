import matplotlib.pyplot as plt
import numpy as np

qubits = np.array([10,12,14,16,18,20,22,24,26,28])

y_un_averages = [1.000000,1.000000,0.988889,0.992593,0.995238,0.997329,0.996198,0.995238,0.995551,0.996010]
y_un_success = [1.000000,1.000000,0.916667,0.933333,0.950000,0.966667,0.966667,0.933333,0.933333,0.933333]

y_we_averages = [0.997802,0.999456,0.996742,0.995035,0.996365,0.997445,0.997444,0.997216,0.996148,0.997126]
y_we_success = [0.933333,0.966667,0.866667,0.850000,0.916667,0.833333,0.866667,0.816667,0.850000,0.800000]

y_re_averages = [0.998306,0.998655,0.999204,0.999645,0.999206,0.997381,0.999046,0.997763,0.998131,0.997644]
y_re_success = [0.950000,0.983333,0.950000,0.983333,0.966667,0.900000,0.916667,0.900000,0.900000,0.850000]

y_er_averages = [1.000000,0.997778,0.996452,1.000000,1.000000,0.997026,0.996530,0.997689,0.996635,0.998231]
y_er_success = [1.000000,0.983333,0.966667,1.000000,1.000000,0.950000,0.933333,0.950000,0.916667,0.950000]

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
plt.ylim(0.983, 1.001)
plt.xticks(qubits)
plt.yticks([0.985,0.990,0.995,1.000])
plt.plot(qubits, y_un_averages, color='red', marker='o', markersize=7,label='unweighted 3-regular')
plt.plot(qubits, y_we_averages, color='blue', marker='d', markersize=7,label='weighted 3-regular')
plt.plot(qubits, y_re_averages, color='green', marker='s', markersize=7,label='weighted 8-regular')
plt.plot(qubits, y_er_averages, color='purple', marker='x', markersize=7,label='Erdos Renyi')
plt.xlabel('Numbers of nodes', fontsize=16)
plt.ylabel('Approximation ratio', fontsize=16)
plt.tight_layout()
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(8, 6), dpi=120, facecolor='lightgray')
bar_width = 0.4
plt.xticks(qubits)
plt.yticks([0.75,1.00])
plt.ylim(0.65, 1.02)
plt.bar(qubits - 1.5 * bar_width, y_un_success, width=bar_width, color='red', edgecolor='black',label='unweighted 3-regular')
plt.bar(qubits - 0.5 * bar_width, y_we_success, width=bar_width, color='blue', edgecolor='black',label='weighted 3-regular')
plt.bar(qubits + 0.5 * bar_width, y_re_success, width=bar_width, color='green', edgecolor='black',label='weighted 8-regular')
plt.bar(qubits + 1.5 * bar_width, y_er_success, width=bar_width, color='purple', edgecolor='black',label='Erdos Renyi')
plt.xlabel('Numbers of nodes', fontsize=16)
plt.ylabel('Success rate', fontsize=16)
plt.tight_layout()
plt.legend(fontsize=9)
plt.show()