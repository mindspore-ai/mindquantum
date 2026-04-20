import os
import pickle
import time
import torch
import numpy as np
point='H2'
pointid=-1
dataname=['F2','H2O','LiH','H2']
runid='layer'
coder='hiq'
runid+=coder
nowtype='r'
if os.path.exists('output/'+runid)==False:
    os.mkdir('output/'+runid)

energy_list=['/mnt/nas-new/home/yange/nature_comm/pack3/energy/F2_dist1.42.pkl','/mnt/nas-new/home/yange/nature_comm/pack3/energy/H2O_0.958_1_104.5.pkl','/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/LiH/energy/LiH_1.pkl','/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/H2/energy/H2_1.pkl']
Operatorpath=['/mnt/nas-new/home/yange/nature_comm/pack3/data/F2_1.42.data','/mnt/nas-new/home/yange/nature_comm/pack3/data/H2O_0.958_1_104.5.data','/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/LiH/data/LiH_1.data','/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/H2/data/H2_1.data']
for id,name in enumerate(dataname):
    if name==point:
        pointid=id
if pointid==-1:
    assert 0
qubits_list=[20,14,12,4]
k_list=[18,10,4,2]
running_list=[]
answer_dict={}

gpu_list = [0,1,2]
running_dict = {gpu: None for gpu in gpu_list}
def check_and_free_gpus():
    for gpu, task in running_dict.items():
        if task is not None and os.path.exists(task):
            running_dict[gpu] = None


# for layers in [1,2,3,4]:
# for layers in [1,2,3,5,10]:
# for layers in [1,2,3,5]:
# for layers in [25,50,100,150,250,350,450]:
# for layers in [25,50,75,125,175,225]:

# for layers in [25,50,100,150,250,350,450,80,90,100,110,120,130,140,150,160,170]:
# for layers in [25,50,75,125,175,225,40,45,50,55,60,65,70,75,80,85]:

# for layers in [1,2,3,4,5,6,7,8,9,10]:
for layers in [1,2,3,4,5]:

# for layers in [10,20,25,30,35,40,50]:
# for layers in [50,60,70,80,90]:
# for layers in [100,110,120,130,140,150,160,170,180]:
# for layers in [50,55,60,65,70,75,80,85,90]:
# for layers in [25,30,35,40,45,50,55]:
# for layers in [5,6,7,8,9,10,11]:
# for layers in [130,150,170,190,210,230,250]:
# for layers in [50,55,60,70,80,90,100,110,120]:
# for layers in [25, 30, 35, 40, 45, 50, 55, 60]:
    answer_dict[layers]=[]
    for randseed in [0,1,2,3,4]:

        Dict={}
        # Dict['--Circuit_type']='fullconnect'
        # Dict['--Circuit_type']='FC'
        # Dict['--Circuit_type']='doublefullconnect'
        Dict['--Circuit_type']='newhea'
        Dict['--Circuit_layers']=layers
        Dict['--Gate_type']="RCNOT"
        # Dict['--Gate_type']="RBS"
        # Dict['--Gate_type']="BS"
        # Dict['--Hamiltonian_path']=hamiltonianpath[pointid]
        if coder=='hiq':
            Dict['--Hamiltonian_path']=Operatorpath[pointid]
        Dict['--Iteration']=10000
        # Dict['--Learningrate']=1e-3
        Dict['--Lowest_energy_path']=energy_list[pointid]
        Dict['--Randseed']=randseed
        Dict['--Qubits_num']=qubits_list[pointid]
        Dict['--K_num']=k_list[pointid]
        Dict['--Output_path']='output/{}/{}/{}_{}_s{}_l{}.pkl'.format(runid,point, Dict['--Gate_type'],Dict['--Circuit_type'],Dict['--Randseed'],layers)
        if os.path.exists('output/{}/{}'.format(runid,point))==False:
            os.mkdir('output/{}/{}'.format(runid,point))
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
            logpath='output/{}/{}/{}_{}_s{}_l{}.log'.format(runid,point, Dict['--Gate_type'],Dict['--Circuit_type'],Dict['--Randseed'],layers)
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

