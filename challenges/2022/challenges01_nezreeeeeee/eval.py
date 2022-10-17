import sys
sys.path.append("./src")
import os
os.environ['OMP_NUM_THREADS'] = '4'
from src.main import Main, Timer
import mindquantum
import numpy as np

# molecule names: H6, LiH, BeH2, NH3, CH4
molecules = ['H6_0.8', 'H6_0.9', 'H6_1.0', #H6
           'LiH_0.4', 'LiH_0.8', 'LiH_1.2', 'LiH_1.6', 'LiH_2.0', 'LiH_2.4', 'LiH_2.8', 'LiH_3.2', 'LiH_3.6', 'LiH_4.0' #LiH
            ]
prefixes = ['H6','H6','H6',
            'LiH','LiH','LiH','LiH','LiH','LiH','LiH','LiH','LiH','LiH'
            ]
fci_en = [-3.2044118794841037, -3.2445422400448987, -3.2360662798923423,
          -6.6402778359670505, -7.63416732932434, -7.852430853195922, -7.882324378883506, -7.861087772481495, -7.8306316244241145, -7.806763402564964, -7.793274300660556, -7.786991815386028, -7.784278178715392
            ]
baseline = [16.0, 16.0, 16.0,
            5., 5., 5., 5., 5., 5., 5., 5., 5., 5.
            ]
err_num = np.ones(len(molecules))

if __name__ == "__main__":
    with open('./output_info.o', 'a') as f:
        main = Main()
        timer = Timer()
        en_list, time, s = [], [], []
        for idx in range(len(molecules)):
            t0 = timer.runtime()
            print('Start: ', molecules[idx], file=f)
            mol_file = './src/hdf5files/'+molecules[idx]+'.hdf5'
            en_list.append(main.run(prefixes[idx], mol_file))
            if (abs(en_list[-1] - fci_en[idx]) <= 0.0016): 
                err_num[idx] = 0
                time.append(timer.runtime()- t0)
            else:
                time.append(3600)
            s.append(time[idx]/baseline[idx])

        if len(en_list) != len(fci_en):
                print('The length of en_list is not equal to that of fci_en !', file=f)

        score = sum(s)
   
        print('Time: ', time, file=f)
        print('Num_err: ', err_num, file=f)
        print('Molecular_names: ', molecules, file=f)
        #print(prefixes, file=f)
        print('FCI_energies: ', fci_en, file=f)
        print('Result_energies: ', en_list, file=f)
        print('Baseline_time: ', baseline, file=f)
        print('Case_scores: ', s, file=f)
        print('Total_Score: ', score, file=f)

        print('Score: ', score)
    