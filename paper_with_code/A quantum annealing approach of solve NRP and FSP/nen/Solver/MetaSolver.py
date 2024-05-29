from typing import Dict, List, Tuple
import time
from cplex import Cplex
from nen.Problem import LP
from nen.Term import Linear, Quadratic
from nen.Result import NDArchive


ObjectiveType = Dict[str, float]


class SolverUtil:
    """ [summary] some useful tools for Solvers.
    """
    @staticmethod
    def scale_objective(obj: ObjectiveType, c: float) -> ObjectiveType:
        """scale_objective [summary] multiply a constant to objective, return it.
        """
        result: ObjectiveType = {}
        for k, v in obj.items(): result[k] = v * c
        return result

    @staticmethod
    def objective_add(o1: ObjectiveType, o2: ObjectiveType) -> None:
        """objective_add [summary] add o2 to o1, inplace operation.
        """
        for k, v in o2.items():
            if k in o1:
                o1[k] += v
            else:
                o1[k] = v

    @staticmethod
    def weighted_sum_objective(objectives: Dict[str, ObjectiveType], weights: Dict[str, float]) -> ObjectiveType:
        """weighted_sum_objective [summary] weighted sum of objectives, given weights for objectives.
        """
        the_obj: ObjectiveType = {}
        for obj_name, weight in weights.items():
            SolverUtil.objective_add(the_obj, SolverUtil.scale_objective(objectives[obj_name], weight))
        return the_obj

    @staticmethod
    def objective_theoretical_boundary(objective: ObjectiveType) -> Tuple[float, float]:
        lb, ub = 0.0, 0.0
        for val in objective.values():
            if val < 0:
                lb += val
            else:
                ub += val
        return (lb, ub)

    @staticmethod
    def normalize(objective: ObjectiveType) -> ObjectiveType:
        """normalize [summary] normalize the objective.
        """
        sum_up = sum(objective.values())
        return {k: v / sum_up for k, v in objective.items()}

    @staticmethod
    def normalized_sum_objective(objectives: Dict[str, ObjectiveType]) -> ObjectiveType:
        """normailized_sum_objective [summary] calculate range of each objective, normalize them, sum up.
        """
        weights = SolverUtil.normalized_weights(objectives)
        return SolverUtil.weighted_sum_objective(objectives, weights)

    @staticmethod
    def normalized_weights(objectives: Dict[str, ObjectiveType]) -> Dict[str, float]:
        """normailized_sum_objective [summary] calculate normalized weights.
        """
        weights: Dict[str, float] = {}
        for name, obj in objectives.items():
            lb, ub = SolverUtil.objective_theoretical_boundary(obj)
            weights[name] = (1 / (ub - lb)) / len(objectives)
        return weights

    @staticmethod
    def scale(objective: ObjectiveType) -> ObjectiveType:
        """scale [summary] scale the objective and let average coefficient is 1.
        """
        sum_up = sum(objective.values()) + 1
        scale_factor = abs(len(objective) / sum_up)
        return {k: v * scale_factor for k, v in objective.items()}

    @staticmethod
    def scaled_weights(objectives: Dict[str, ObjectiveType]) -> Dict[str, float]:
        """scaled_weights [summary] scale objectives and let average of coefs are 1.
        """
        weights: Dict[str, float] = {}
        for name, objective in objectives.items():
            sum_up = sum(objective.values()) + 1
            weights[name] = abs(len(objective) / sum_up)
        return weights

    @staticmethod
    def collect_quadratic_coefs(quadratic: Quadratic) -> Tuple[List[float], List[float]]:
        """collect_quadratic_coefs [summary] collect coefs in a quadratic, return hs and Js.
        """
        return list(quadratic.linear.values()), list(quadratic.quadratic.values())

    @staticmethod
    def max_gradient(coefs: List[float]) -> float:
        """max_gradient [summary] return the max gradient in a list of coefs.
        """
        return max([abs(c) for c in coefs])

    @staticmethod
    def min_gradient(coefs: List[float]) -> float:
        """min_gradient [summary] return the min gradient in a list of coefs.
        """
        return min([abs(c) for c in coefs])

    @staticmethod
    def time() -> float:
        """time [summary] return time for count elapsed second(s).
        """
        return time.perf_counter()


class ExactSolver:
    """ [summary] ExactSolver is a meta-solver for all exact solving met,
    it defines some useful operations for using cplex to solve a lp.
    """
    # configs
    # cplex threads, 0 for auto
    CPLEX_THREADS = 1

    def __init__(self, problem: LP) -> None:
        """__init__ [summary] implement a cplex solver and initialize it.
        """
        self.solver: Cplex = ExactSolver.initialized_cplex_solver(problem)

    @staticmethod
    def initialized_cplex_solver(problem: LP) -> Cplex:
        """initialized_cplex_solver [summary] initialize a cplex solver with variables and constraints.
        """
        # prepare cplex and set variables
        solver = ExactSolver.variables_initialized_cplex_solver(problem.variables)
        # add constraints
        for index, constraint in enumerate(problem.constraints_lp):
            ExactSolver.add_constraint(solver, 'c{}'.format(index), constraint)
        # return
        return solver

    @staticmethod
    def variables_initialized_cplex_solver(variables: List[str]) -> Cplex:
        """variables_initialized_cplex_solver [summary] initialize a cplex with variables,
        left objectives and constraints unset.
        """
        # prepare cplex
        solver: Cplex = Cplex()
        solver.set_results_stream(None)
        solver.set_warning_stream(None)
        solver.set_error_stream(None)
        solver.parameters.threads.set(ExactSolver.CPLEX_THREADS)
        solver.parameters.emphasis.mip.set(0)
        solver.parameters.mip.tolerances.absmipgap.set(0.0)
        solver.parameters.mip.tolerances.mipgap.set(0.0)
        # solver.parameters.timelimit.set(10000)
        # solver.parameters.dettimelimit.set(10000)
        # add variables
        solver.variables.add(obj=None, lb=None, ub=None, types='B' * len(variables), names=variables, columns=None)
        return solver

    @staticmethod
    def set_minimizing_objective(solver: Cplex, objective: Dict[str, float]) -> None:
        """set_minimizing_objective [summary] set linear objective with sense 'minimize'.
        """
        solver.objective.set_linear([(k, v) for k, v in objective.items()])
        solver.objective.set_sense(solver.objective.sense.minimize)

    @staticmethod
    def set_minimizing_qudratic_objective(solver: Cplex, objective: Quadratic) -> None:
        """set_minimizing_qudratic_objective [summary] set quadratic objective with sense 'minimize'.
        """
        for k, v in objective.linear.items():
            solver.objective.set_linear(k, v)
        for (k1, k2), v in objective.quadratic.items():
            solver.objective.set_quadratic_coefficients(k1, k2, v)
        solver.objective.set_sense(solver.objective.sense.minimize)

    @staticmethod
    def solve(solver: Cplex) -> None:
        """solve [summary] just solve.
        """
        solver.solve()

    @staticmethod
    def solve_and_count(solver: Cplex) -> float:
        """solve_and_count[summary] solve the problem and return the elapsed time (second(s)).
        """
        start = time.perf_counter()
        solver.solve()
        end = time.perf_counter()
        return end - start

    @staticmethod
    def solve_and_get_values(solver: Cplex, variables: List[str]) -> Dict[str, bool]:
        """solve_and_get_values [summary] solve and get variables values, return {} if not optimal.
        """
        solver.solve()
        return ExactSolver.get_values(solver, variables)

    @staticmethod
    def get_values(solver: Cplex, variables: List[str]) -> Dict[str, bool]:
        """get_values [summary] get variables values after once solving, return {} if not optimal.
        """
        if 'optimal' not in solver.solution.get_status_string():
            return {}
        else:
            values: Dict[str, bool] = {}
            for var in variables:
                # TODO: maybe == 1 instead of > 0.5?
                values[var] = (solver.solution.get_values(var) > 0.5)
            return values

    @staticmethod
    def add_constraint(solver: Cplex, name: str, constraint: Linear) -> None:
        var = list(constraint.coef.keys())
        val = [constraint.coef[v] for v in var]
        sense = 'L' if constraint.sense == '<=' else 'E'
        solver.linear_constraints.add(lin_expr=[[var, val]], senses=sense,
                                      rhs=[constraint.rhs], names=[name])

    @staticmethod
    def set_constraint_rhs(solver: Cplex, name: str, rhs: float) -> None:
        solver.linear_constraints.set_rhs(name, rhs)

    @staticmethod
    def epsilon_constraint_recurse(solver: Cplex,
                                   current: int,
                                   obj_names: List[str],
                                   obj_boundaries: List[Tuple[float, float]],
                                   step: float,
                                   variables: List[str],
                                   values_list: List[Dict[str, bool]]) -> None:
        """epsilon_constraint_recurse [summary] set current objective constraint rhs and recurse,
        if current == -1 then solve. And if there's a solution, collect it.
        """
        if current == -1:
            values = ExactSolver.solve_and_get_values(solver, variables)
            if values == {}: return
            values_list.append(values)
        else:
            obj_name = obj_names[current]
            lb, ub = obj_boundaries[current]
            rhs = ub
            while rhs >= lb:
                ExactSolver.set_constraint_rhs(solver, obj_name, rhs)
                ExactSolver.epsilon_constraint_recurse(solver, current - 1, obj_names, obj_boundaries,
                                                       step, variables, values_list)
                rhs -= step

    @staticmethod
    def epsilon_constraint(problem: LP, step: float) -> Tuple[NDArchive, float]:
        # prepare a solver
        solver = ExactSolver.initialized_cplex_solver(problem)
        # calculate theoretical boundaries of objectives[1:] and add constraint into solver
        reduced_names: List[str] = []
        boundaries: List[Tuple[float, float]] = []
        for obj_name in problem.objectives_order[1:]:
            obj = problem.objectives[obj_name]
            reduced_names.append(obj_name)
            boundaries.append(SolverUtil.objective_theoretical_boundary(obj))
            ExactSolver.add_constraint(solver, obj_name, Linear(obj, '<=', 0.0))
        # set objectives
        ExactSolver.set_minimizing_objective(solver, problem.objectives[problem.objectives_order[0]])
        # solve
        values_list: List[Dict[str, bool]] = []
        start = time.perf_counter()
        ExactSolver.epsilon_constraint_recurse(solver, len(reduced_names) - 1, reduced_names, boundaries,
                                               step, problem.variables, values_list)
        end = time.perf_counter()
        # prepare an archive
        archive = NDArchive(problem.variables_num, problem.objectives_num)
        # evaluate solutions and add into archive
        for values in values_list:
            archive.add(problem.evaluate(values))
        # return
        return archive, end - start
