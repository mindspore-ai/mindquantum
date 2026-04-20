import os
import pickle
import time
import numpy as np

point = 'LiH'
GPU = 0
pointid = -1
dataname = ['F2', 'BeH2', 'LiH', 'H2', 'H2O']
energy_list = ['/mnt/nas-new/home/yange/nature_comm/pack3/energy/F2', '/mnt/nas-new/home/yange/nature_comm/pack2/energy/BeH2',
               '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/LiH/energy', '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/H2/energy', '/mnt/nas-new/home/yange/nature_comm/pack3/energy/H2O']
Operatorpath = ['/mnt/nas-new/home/yange/nature_comm/pack3/data/F2', '/mnt/nas-new/home/yange/nature_comm/pack2/data/BeH2',
                '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/LiH/data', '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/H2/data',
                '/mnt/nas-new/home/yange/nature_comm/pack3/data/H2O']

for id, name in enumerate(dataname):
    if name == point:
        pointid = id
if pointid == -1:
    raise AssertionError("Point not found in dataname list.")

qubits_list = [20, 14, 12, 4, 14]
k_list = [18, 6, 4, 2, 10]
bound_dist = [1.42, 1.33, 1, 1, 0.958]

runid = 'circuit'
coder = 'UCC'
runid += coder
nowtype = 'r'

# Ensure output directory exists
output_dir = 'output/' + runid
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

running_list = []
answer_dict = {}
# for angle in [40, 60, 80, 90, 100, 104.5, 110, 120, 150, 180]:
#     for factor in [0.7, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3, 1.5]:
# # for angle in [40]:
#     # for factor in [1.3]:
#         for seed in range(10):  # Run for 5 random seeds
#             Dict = {
#                 '--Hamiltonian_path': f'{Operatorpath[pointid]}/{point}_{bound_dist[pointid]}_{factor}_{angle}.data',
#                 '--Lowest_energy_path': f'{energy_list[pointid]}/{point}_{bound_dist[pointid]}_{factor}_{angle}.pkl',
#                 '--n_qubits': qubits_list[pointid],
#                 '--n_electrons': k_list[pointid],
#                 '--Randseed': seed
#             }

#             Dict['--Output_path'] = f'output/{runid}/{point}_{factor}_{angle}_seed{seed}.pkl'

#             if nowtype == 'r':
#                 # Construct command to run UCCSD.py script
#                 command = f'CUDA_VISIBLE_DEVICES={GPU} nohup python -u UCCSD.py '
#                 for key, value in Dict.items():
#                     command += f' {key} {value} '
#                 logpath = f'output/{runid}/{point}_{factor}_{angle}_seed{seed}.log'
#                 if os.path.exists(logpath):
#                     continue  # Skip if log already exists
#                 command += f'> {logpath} 2>&1 &'
#                 print(command)
#                 os.system(command)

#                 if coder in ['UCC', 'hiq']:
#                     running_list.append(Dict['--Output_path'])
#                     while len(running_list) >= 1:
#                         for i in range(len(running_list)):
#                             time.sleep(5)
#                             if os.path.exists(running_list[i]):
#                                 del running_list[i]
#                                 break
#                         print("Waiting For Result")
#                 time.sleep(0.1)
            
#             elif nowtype == 'a':
#                 answer_0 = pickle.load(open(f'output/{runid}/{point}_{factor}_{angle}_seed{seed}.pkl', 'rb'))
#                 answer_1 = pickle.load(open(Dict['--Output_path'], 'rb'))
#                 real = pickle.load(open(Dict['--Lowest_energy_path'], 'rb'))
#                 answer = min(np.min(answer_0[-1]), np.min(answer_1[-1]))
#                 print(answer - real)
#                 # answer_dict[layers].append(answer - real)


# for factor in [0.7,0.9,0.95,1,1.05,1.1,1.3,1.6,2,3]:
#     for seed in range(10):  # Run for 5 random seeds
#         Dict = {

#             # '--Hamiltonian_path':  f'{Operatorpath[pointid]}/{point}_{bound_dist[pointid]}_{factor}.data',
#             # '--Lowest_energy_path': f'{energy_list[pointid]}/{point}_{bound_dist[pointid]}_{factor}.pkl',
#             '--Hamiltonian_path':  f'{Operatorpath[pointid]}/{point}_{factor}.data',
#             '--Lowest_energy_path': f'{energy_list[pointid]}/{point}_{factor}.pkl',
#             '--n_qubits': qubits_list[pointid],
#             '--n_electrons': k_list[pointid],
#             '--Randseed': seed
#         }

#         Dict['--Output_path'] = f'output/{runid}/{point}_{factor}_seed{seed}.pkl'

#         if nowtype == 'r':
#             # Construct command to run UCCSD.py script
#             command = f'CUDA_VISIBLE_DEVICES={GPU} nohup python -u UCCSD.py '
#             for key, value in Dict.items():
#                 command += f' {key} {value} '
#             logpath = f'output/{runid}/{point}_{factor}_seed{seed}.log'
#             if os.path.exists(logpath):
#                 continue  # Skip if log already exists
#             command += f'> {logpath} 2>&1 &'
#             print(command)
#             os.system(command)

#             if coder in ['UCC', 'hiq']:
#                 running_list.append(Dict['--Output_path'])
#                 while len(running_list) >= 1:
#                     for i in range(len(running_list)):
#                         time.sleep(5)
#                         if os.path.exists(running_list[i]):
#                             del running_list[i]
#                             break
#                     print("Waiting For Result")
#             time.sleep(0.1)
        
#         elif nowtype == 'a':
#             answer_0 = pickle.load(open(f'output/{runid}/{point}_{factor}_{angle}_seed{seed}.pkl', 'rb'))
#             answer_1 = pickle.load(open(Dict['--Output_path'], 'rb'))
#             real = pickle.load(open(Dict['--Lowest_energy_path'], 'rb'))
#             answer = min(np.min(answer_0[-1]), np.min(answer_1[-1]))
#             print(answer - real)
#             # answer_dict[layers].append(answer - real)



# for factor in [0.7,0.9,0.95,1,1.05,1.1,1.3,1.6,2,3]:  # [0.7,0.9,0.95,1,1.05,1.1,1.3,1.6,2]:
for factor in [1]:    
    Dict = {}
    # Dict['--Hamiltonian_path'] = '{}/{}_{}_{}.data'.format(Operatorpath[pointid], point, bound_dist[pointid],
    #                                                        factor)
    # Dict['--Lowest_energy_path'] = '{}/{}_{}_{}.pkl'.format(energy_list[pointid], point, bound_dist[pointid],
    #                                                         factor)
    Dict['--Hamiltonian_path'] = '{}/{}_{}.data'.format(Operatorpath[pointid], point,
                                                           factor)
    Dict['--Lowest_energy_path'] = '{}/{}_{}.pkl'.format(energy_list[pointid], point,
                                                            factor)
    Dict['--n_qubits'] = qubits_list[pointid]
    Dict['--n_electrons'] = k_list[pointid]
    Dict['--Output_path'] = 'output/{}/{}_{}.pkl'.format(runid, point, factor)
    if nowtype == 'r':
        str = 'CUDA_VISIBLE_DEVICES={} nohup python -u UCCSD.py '.format(GPU)
        for i in Dict.keys():
            str += ' {} {} '.format(i, Dict[i])
        logpath = 'output/{}/{}_{}.log'.format(runid, point, factor)
        if os.path.exists(logpath) == True:
            # assert 0
            continue
        str += '> {} 2>&1 &'.format(logpath)
        print(str)
        os.system(str)
        if coder == 'UCC' or coder == 'hiq':
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
        # answer_dict[layers].append(answer[-1][-1] - real)
        # answer_dict[layers].append(answer[-1][-1])
        # exit(0)
    else:
        assert 0