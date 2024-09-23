from Figure1_cd_functions import *

'''
    一共20个参数，当至少有18个在区间 [0.78 +- 0.05, 0 +- 0.05, -0.78 +- 0.05] 的时候才执行绘图函数
    当绘制50个图的时候break
    对四种分布方式的图都执行，文献只要求 uniform 和 normal
    circle_func 中的 distribution 按需修改
'''
intervals = [(0.63, 0.83), (-0.83, -0.63), (-0.05, 0.05)]

def circle_func(num, distribution):
    va_all = []

    for i in range(0, num):
        index = 0
        qubo = graph_complete(6, distribution)
        pool = mixer_pool_single(6) + mixer_pool_multi(6)
        pr, va, _ = ADAPT_QAOA(6, qubo, pool, 10, 'bfgs')

        pr_list = list(pr.values())
        gama = [pr_list[x] for x in range(0, 20, 2)]
        beta = [pr_list[x] for x in range(1, 20, 2)]

        for x in gama:
            if any(lower <= x <= upper for lower, upper in intervals):
                index += 1
        for y in beta:
            if any(lower <= y <= upper for lower, upper in intervals):
                index += 1

        print(f'This is {i} th graph')
        if index >= 18:
            draw_pr(gama, beta, f'distribution/distribution_{i}')

            va_all.append(va[-1])

        if len(va_all) == 50:
            break

circle_func(10000)