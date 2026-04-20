import os
import pickle
import time
import torch
import numpy as np
point='F2'
pointid=-1
dataname=['F2','BeH2','LiH','H2']
energy_list=['/home/yange/nature_comm/pack3/energy/F2_dist1.42.pkl','/home/yange/nature_comm/pack3/energy/BeH2_dist1.34.pkl','/home/yange/nature_comm/pack3/data/31g/LiH_1.595.pkl','/home/yange/nature_comm/pack3/data/31g/H2_1.2.pkl']
Operatorpath=['/home/yange/nature_comm/pack3/data/F2_dist1.42.data','/home/yange/nature_comm/pack3/data/BeH2_dist1.34.data','/home/yange/nature_comm/pack3/data/31g/LiH_1.595.data','/home/yange/nature_comm/pack3/data/31g/H2_1.2.data']
for id,name in enumerate(dataname):
    if name==point:
        pointid=id
if pointid==-1:
    assert 0
qubits_list=[20,14,22,8]
k_list=[18,6,4,2]

runid='test'
coder='hiq'
runid+=coder
nowtype='r'
if os.path.exists('output/'+runid)==False:
    os.mkdir('output/'+runid)
running_list=[]
answer_dict={}
for layers in [60]:
    answer_dict[layers]=[]
    for randseed in [0,1]:
        Dict={}
        # Dict['--Circuit_type']='fullconnect'
        Dict['--Circuit_type']='FC'
        # Dict['--Circuit_type']='doublefullconnect'
        # Dict['--Circuit_type']='newhea'
        Dict['--Circuit_layers']=layers
        # Dict['--Gate_type']="RCNOT"
        Dict['--Gate_type']="RBS"
        # Dict['--Gate_type']="BS"
        # Dict['--Hamiltonian_path']=hamiltonianpath[pointid]
        if coder=='hiq':
            Dict['--Hamiltonian_path']=Operatorpath[pointid]
        Dict['--Iteration']=10000
        # Dict['--Learningrate']=1e-3
        if energy_list[pointid]!=None:
            Dict['--Lowest_energy_path']=energy_list[pointid]
        Dict['--Randseed']=randseed
        Dict['--Qubits_num']=qubits_list[pointid]
        Dict['--K_num']=k_list[pointid]
        Dict['--Output_path']='output/{}/test_RBS.pkl'.format(runid)
        
        if nowtype=='r':
            str='CUDA_VISIBLE_DEVICES=0 nohup python -u ./{}/base.py '.format(coder)
            
            for i in Dict.keys():
                str+=' {} {} '.format(i,Dict[i])
            logpath='output/{}/test_RBS.log'.format(runid)
            if os.path.exists(logpath)==True:
                # assert 0
                continue
            str+='> {} 2>&1 &'.format(logpath)
            print(str)
            os.system(str)
            if coder=='pks' or coder=='hiq':
                running_list.append(Dict['--Output_path'])
                # while len(running_list)>=1:
                #     for i in range(len(running_list)):
                #         time.sleep(5)
                #         if os.path.exists(running_list[i])==True:
                #             del running_list[i]
                #             break
                    # print("Waiting For Result")
            time.sleep(0.1)
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

