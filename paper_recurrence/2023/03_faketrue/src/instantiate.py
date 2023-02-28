import numpy as np

# 映射函数实例化参数
def map_para(n,r):
    para = {}
    for i in range(n):
        para[f'theta{i}'] = (2*np.pi/((i+1)*r)-np.pi)
    return para



# 随机实例化参数
def random_para(n):
    para = {}
    for i in range(n):
        para[f'theta{i}'] = (np.random.uniform(np.pi,-np.pi))
    return para



# 实例化参数验证circ1和circ2，验证r轮
def verify_by_para(circ1,circ2,r):
    # 线路中一共n个参数
    n = len(list(set(circ1.params_name+circ2.params_name)))
    # 记录前r-1轮验证是否有结果
    flag = True
    # 前r-1轮指定参数
    for i in range(r-1):
        para = map_para(n,i+1)

        if np.array_equal(circ1.matrix(para), circ2.matrix(para)):
            continue
        else:
            print('Not equivalent!')
            flag = False
            break

    # 最后一轮随机参数
    if flag:
        para = random_para(n)
        if np.array_equal(circ1.matrix(para), circ2.matrix(para)):
            print('Equivalent!')
        else:
            print('Not equivalent!')





