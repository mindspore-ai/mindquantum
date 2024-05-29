import sys
import os
from nen import Problem, ProblemResult, MethodResult, QP
from nen.Solver.HybridSolver import HybridSolver

# 将NEN库的路径添加到系统路径中
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

def evaluate_hybrid_performance(problems, sizes):
    results = {}

    for problem_name, order in problems.items():
        # 创建问题对象
        problem = QP(problem_name)
        problem.vectorize(order)

        for size in sizes:
            # 创建问题结果对象
            problem_result = ProblemResult(problem_name, problem, f'hybrid_evaluation_{size}')
            hy_result = MethodResult(f'hybrid{size}', problem_result.path, problem)

            # 尝试用HybridSolver求解问题
            try:
                result = HybridSolver.solve(problem=problem, sample_times=10, num_reads=100,
                                            sub_size=size, annealing_time=20)
                hy_result.add(result)
                problem_result.add(hy_result)
                problem_result.dump()

                # 记录性能数据
                performance_data = {
                    'elapsed_time': result.elapsed,
                    'iterations': result.iterations,
                    'total_num_anneals': result.total_num_anneals,
                    # 假设 result 中有一个 best_solution_quality 属性
                    'best_solution_quality': result.best_solution_quality if hasattr(result, 'best_solution_quality') else None
                }
                results[(problem_name, size)] = performance_data
                print
                print(f"成功解决了大小为 {size} 的 {problem_name} 问题。")
            except Exception as e:
                print(f"无法解决大小为 {size} 的 {problem_name} 问题：{e}")
                break

    return results

# 示例问题定义
problems = {
    'uClinux': ['COST', 'USED_BEFORE', 'DEFECTS', 'DESELECTED'],
    'classic-4': ['cost', 'revenue']
}
sizes = [100, 500, 700, 1000, 1200]

performance_results = evaluate_hybrid_performance(problems, sizes)
print(performance_results)
