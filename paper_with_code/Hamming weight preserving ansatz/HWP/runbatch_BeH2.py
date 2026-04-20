import os
import pickle
import time
import torch
import numpy as np

point = 'BeH2'
GPU = 2
pointid = -1
dataname = ['F2', 'BeH2', 'LiH']
energy_list = ['/home/yange/nature_comm/pack3/energy/F2_dist1.42.pkl', '/data1/yange/pack3/energy/BeH2_1.33.pkl',
               '/home/yange/nature_comm/pack3/energy/LiH.pkl']
Operatorpath = ['/home/yange/nature_comm/pack3/data/F2_1.42.data', '/data1/yange/pack3/data/BeH2/BeH2_1.33.data',
                '/home/yange/nature_comm/pack3/data/LiH_12_4.data']
for id, name in enumerate(dataname):
    if name == point:
        pointid = id
if pointid == -1:
    assert 0
qubits_list = [20, 14, 12]
k_list = [18, 6, 4]

runid = 'test'
coder = 'hiq'
runid += coder
nowtype = 'r'
if os.path.exists('output/' + runid) == False:
    os.mkdir('output/' + runid)
running_list = []
answer_dict = {}
for layers in [150, 160, 170]:
    answer_dict[layers] = []
    for randseed in [42]:
        Dict = {}
        # Dict['--Circuit_type']='fullconnect'
        # Dict['--Circuit_type']='doublefullconnect'
        Dict['--Circuit_type'] = 'newhea'
        Dict['--Circuit_layers'] = layers
        # Dict['--Gate_type']="RCNOT"
        # Dict['--Gate_type']="RBS"
        Dict['--Gate_type'] = "BS"
        # Dict['--Hamiltonian_path']=hamiltonianpath[pointid]
        if coder == 'hiq':
            Dict['--Hamiltonian_path'] = Operatorpath[pointid]
        Dict['--Iteration'] = 10000
        Dict['--Learningrate'] = 5e-2
        if energy_list[pointid] != None:
            Dict['--Lowest_energy_path'] = energy_list[pointid]
        Dict['--Randseed'] = randseed
        Dict['--Qubits_num'] = qubits_list[pointid]
        Dict['--K_num'] = k_list[pointid]
        Dict['--Output_path'] = 'output/{}/{}_{}layers_{}_seed{}_{}.pkl'.format(runid, point, layers,
                                                                                Dict['--Gate_type'], Dict['--Randseed'],
                                                                                Dict['--Circuit_type'])

        if nowtype == 'r':
            str = 'CUDA_VISIBLE_DEVICES={} nohup python -u ./{}/base.py '.format(GPU, coder)

            for i in Dict.keys():
                str += ' {} {} '.format(i, Dict[i])
            logpath = 'output/{}/{}_{}layers_{}_seed{}_{}.log'.format(runid, point, layers, Dict['--Gate_type'],
                                                                      Dict['--Randseed'], Dict['--Circuit_type'])
            if os.path.exists(logpath) == True:
                # assert 0
                continue
            str += '> {} 2>&1 &'.format(logpath)
            print(str)
            os.system(str)
            if coder == 'pks' or coder == 'hiq':
                running_list.append(Dict['--Output_path'])
                while len(running_list) >= 1:
                    for i in range(len(running_list)):
                        time.sleep(5)
                        if os.path.exists(running_list[i]) == True:
                            del running_list[i]
                            break
                    print("Waiting For Result")
            time.sleep(0.1)
        elif nowtype == 'a':
            answer = pickle.load(open(Dict['--Output_path'], 'rb'))
            real = pickle.load(open(energy_list[pointid], 'rb'))
            # print(answer[-1][-1])
            print(answer)
            answer_dict[layers].append(answer[-1][-1] - real)
            # answer_dict[layers].append(answer[-1][-1])
            # exit(0)
        else:
            assert 0
    # print(answer_dict)
    answer_dict[layers] = sorted(answer_dict[layers])
    print(answer_dict[layers])
    print(np.mean(answer_dict[layers][:10]))
    print(np.std(answer_dict[layers][:10]))

