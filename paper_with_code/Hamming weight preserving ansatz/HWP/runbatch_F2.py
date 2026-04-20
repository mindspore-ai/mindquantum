import os
import pickle
import time
import torch
import numpy as np
point='F2'
runid='new_test'
coder='hiq'
runid+=coder
nowtype='r'
if os.path.exists('output/'+runid)==False:
    os.mkdir('output/'+runid)
running_list=[]
answer_dict={}
factors = [0.7, 0.9, 0.95, 1, 1.05, 1.1, 1.3, 1.6,2]
# factors = [0.7, 0.9, 0.95, 1, 1.05, 1.1, 1.3, 1.6,2,3]
# factors = [0.7, 0.9, 0.95, 1, 1.05,1.1]
# factors = [1.1]
best_dist = 1.42
gpu_list = [0,1]
# gpu_list = [1,2,4]
running_dict = {gpu: None for gpu in gpu_list}
def check_and_free_gpus():
    for gpu, task in running_dict.items():
        if task is not None and os.path.exists(task):
            running_dict[gpu] = None

# for layers in [200]:
for layers in [100]:
    answer_dict[layers]=[]
    for factor in factors:
        for randseed in [0,1,2,3,4]:

            Dict={}
            # Dict['--Circuit_type']='fullconnect'
            # Dict['--Circuit_type']='doublefullconnect'
            Dict['--Circuit_type']='newhea'
            Dict['--Circuit_layers']=layers
            Dict['--Gate_type']="RCNOT"
            # Dict['--Gate_type']="RBS"
            # Dict['--Gate_type']="BS"
            # Dict['--Hamiltonian_path']=hamiltonianpath[pointid]
            if coder=='hiq':
                Dict['--Hamiltonian_path']='/mnt/nas-new/home/yange/nature_comm/pack3/data/{}/{}_{}_{}.data'.format(point,point,best_dist,factor)
            Dict['--Iteration']=10000
            Dict['--Learningrate']=1e-4
            Dict['--Lowest_energy_path']='/mnt/nas-new/home/yange/nature_comm/pack3/energy/{}/{}_{}_{}.pkl'.format(point,point,best_dist,factor)
            Dict['--Randseed']=randseed
            Dict['--Qubits_num']=20
            Dict['--K_num']=18
            Dict['--Output_path']='output/{}/{}/{}_{}_s{}_l{}.pkl'.format(runid,point,factor,Dict['--Gate_type'],Dict['--Randseed'],layers)
            
            if nowtype == 'r':
                assigned_gpu = None
                while assigned_gpu is None:
                    check_and_free_gpus()
                    for gpu, task in running_dict.items():
                        if task is None:
                            assigned_gpu = gpu
                            break
                    if assigned_gpu is None:
                        time.sleep(1)
                str = f'CUDA_VISIBLE_DEVICES={assigned_gpu} nohup python -u ./{coder}/base.py '
                
                for i in Dict.keys():
                    str+=' {} {} '.format(i,Dict[i])
                logpath='output/{}/{}/{}_{}_s{}_l{}.log'.format(runid,point,factor,Dict['--Gate_type'],Dict['--Randseed'],layers)
                if os.path.exists(logpath)==True:
                    # assert 0
                    continue
                str+='> {} 2>&1 &'.format(logpath)
                print(str)
                os.system(str)
                running_dict[assigned_gpu] = Dict['--Output_path']
                time.sleep(0.1)

                # if coder=='pks' or coder=='hiq':
                #     running_list.append(Dict['--Output_path'])
                #     while len(running_list)>=2:
                #         for i in range(len(running_list)):
                #             time.sleep(5)
                #             if os.path.exists(running_list[i])==True:
                #                 del running_list[i]
                #                 break
                #         print("Waiting For Result")
                # time.sleep(0.1)
            elif nowtype=='a':
                answer=pickle.load(open(Dict['--Output_path'],'rb'))
                real=pickle.load(open(energy_list[pointid],'rb'))
                # print(answer[-1][-1])
                print(answer)
                answer_dict[layers].append(answer[-1][-1]-real)
                # answer_dict[layers].append(answer[-1][-1])
                # exit(0)
            else: assert 0
    # print(answer_dict)
    answer_dict[layers] = sorted(answer_dict[layers])
    print(answer_dict[layers])
    print(np.mean(answer_dict[layers][:10]))
    print(np.std(answer_dict[layers][:10]))

