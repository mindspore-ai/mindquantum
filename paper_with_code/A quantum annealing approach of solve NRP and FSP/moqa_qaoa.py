import sys
import os
from nen import QP, ProblemResult, MethodResult
from nen.Solver.MOQASolverQAOA import MOQASolverQAOA  # 确保你已经正确导入了 MOQASolverQAOA 类
from nen.Solver.MOQASolverQAOA_MQ import MOQASolverQAOA_MQ

# 获取当前路径和根路径
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# 定义问题的名称和顺序
names_FSP = ['uClinux', 'BerkeleyDB']
order_FSP = ['COST', 'USED_BEFORE', 'DEFECTS', 'DESELECTED']

names_NRP = ['rp', 'ms']
order_NRP = ['cost', 'revenue']

# 结果保存的文件夹
result_folder = 'moqa_qaoa'

# 解决 NRP 问题
# for name in names_NRP:
#     problem = QP(name, order_NRP)
#     problem_result = ProblemResult(name, problem, result_folder)
#     moqa_method_result = MethodResult('moqa_qaoa', problem_result.path, problem)
    
#     for _ in range(3):
#         result = MOQASolverQAOA.solve(problem=problem, sample_times=10, num_reads=100)
#         moqa_method_result.add(result)

#     # 将结果添加到方法结果和问题结果中
#     problem_result.add(moqa_method_result)

#     # 将结果保存到指定文件夹中
#     problem_result.dump()

# 解决 FSP 问题
for name in names_FSP:
    problem = QP(name, order_FSP)
    problem_result = ProblemResult(name, problem, result_folder)
    moqa_method_result = MethodResult('moqa_qaoa', problem_result.path, problem)
    
    for _ in range(3):
        result = MOQASolverQAOA_MQ.solve(problem=problem, sample_times=10, num_reads=100)
        moqa_method_result.add(result)

    # 将结果添加到方法结果和问题结果中
    problem_result.add(moqa_method_result)

    # 将结果保存到指定文件夹中
    problem_result.dump()