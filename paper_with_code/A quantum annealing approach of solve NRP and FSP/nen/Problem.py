from typing import List, Dict, Tuple
import copy
from numpy import matrix as Mat
from jmetal.core.solution import BinarySolution
from nen.DescribedProblem import DescribedProblem
from nen.Term import Constraint, Linear, Quadratic
from pymoo.core.problem import Problem as Pro
import numpy as np


class Problem:
    """ [summary] Problem is the logic problem used in this project, its name determines its described problem file.

    Note that, 'Problem' is composed by LINEAR objectives and INTEGER (coefficient) constraints.
    """
    def __init__(self, name: str = '', offset_flag: bool = False) -> None:
        # record the name
        self.constrains_index: Dict[str, int] = {}
        self.name: str = name
        # problem content
        self.variables: List[str] = []
        self.objectives: Dict[str, Dict[str, float]] = {}
        self.each_objevtive_sum: List[float] = []
        # self.scaled_objectives: Dict[str, Dict[str, float]] = {}
        self.constraints: List[Constraint] = []
        self.variables_num: int = 0
        self.objectives_num: int = 0
        self.constraints_num: int = 0
        self.eq_constraints_num: int = 0
        self.ieq_constraints_num: int = 0

        # evaluate all constraints or just indicate infeasible (1)
        self.violateds_count: bool = True
        self.offset_flag: bool = offset_flag

        # vectorized content
        self.variables_index: Dict[str, int] = {var: index for index, var in enumerate(self.variables)}
        self.all_variables_index: Dict[str, int] = {var: index for index, var in enumerate(self.variables)}
        self.objectives_order: List[str] = []
        self.objectives_index: Dict[str, int] = {}
        self.objectives_matrix: Mat = Mat([])

        # check if empty problem
        if name == '': return

        # load with described problem
        dp = DescribedProblem()
        dp.load(name)
        self.variables = dp.variables
        self.objectives = dp.objectives.copy()
        # self.scaled_objectives = self.objectives.copy()
        self.each_objevtive_sum = [0.0 for _ in range(len(self.objectives))]
        i = 0
        for k, v in dp.objectives.items():
            for var in dp.variables:
                self.objectives[k][var] = v[var] if var in v else 0.0
                # self.scaled_objectives[k][var] = v[var] if var in v else 0.0
                self.each_objevtive_sum[i] += v[var] if var in v else 0.0
                # if offset_flag and self.objectives[k][var] < 0:
                #     self.objectives[k][var] *= 10
                # if offset_flag:
                #     self.offset_objectives[k][var] *= 2
            i += 1
        assert i == len(self.objectives)

        # j = 0
        # for k, v in dp.objectives.items():
        #     for var in dp.variables:
        #         self.scaled_objectives[k][var] = v[var] / self.each_objevtive_sum[j] if var in v else 0.0
        #     j += 1
        # assert j == len(self.objectives)
        

        for constraint_str_list in dp.constraints:
            assert len(constraint_str_list) == 3
            left, sense, right = constraint_str_list
            if sense == '=' or sense == '<=>':
                self.eq_constraints_num += 1
            else:
                self.ieq_constraints_num += 1
            if sense == '<=' or sense == '=':
                # self.constraints.append(Constraint(left, sense, int(right)))
                self.constraints.append(Constraint(left, sense, right))
            else:
                self.constraints.append(Constraint(left, sense, right))
        self.variables_num = len(self.variables)
        self.objectives_num = len(self.objectives)
        self.constraints_num = len(self.constraints)
        assert self.ieq_constraints_num + self.eq_constraints_num == self.constraints_num, "ieq_constraints_num + eq_constraints_num is wrong!"

    def info(self) -> None:
        print('name: {}'.format(self.name))
        print('variables number: {}'.format(len(self.variables)))
        print('objectives: {}'.format(self.objectives_order))
        print('constraints number: {}'.format(len(self.constraints)))

    def clone(self, another: 'Problem') -> None:
        """clone [summary] construct from another problem type
        """
        self.name = another.name
        self.variables = copy.copy(another.variables)
        self.objectives = copy.deepcopy(another.objectives)
        self.constraints = copy.deepcopy(another.constraints)

    def vectorize_variables(self) -> None:
        """vectorize_variables [summary] index variables
        """
        for index, var in enumerate(self.variables):
            self.variables_index[var] = index

    def vectorize_objectives(self, objectives_order: List[str]) -> None:
        """vectorize_objectives [summary] make objectives a matrix (in numpy) to accelerate speed.
        """
        assert self.variables_index is not None
        # collect objectives and check if objectives order is legal
        self.objectives_index = {}
        ordered_objectives: List[List[float]] = [[0.0] * len(self.variables) for _ in range(len(objectives_order))]
        for index, objective_name in enumerate(objectives_order):
            self.objectives_index[objective_name] = index
            assert objective_name in self.objectives
            for var, coef in self.objectives[objective_name].items():
                ordered_objectives[index][self.variables_index[var]] = coef
        self.objectives_matrix = Mat(ordered_objectives).T
        # set objectives attributes
        self.objectives_order = objectives_order
        self.objectives_num = len(objectives_order)
        self.objectives = {name: self.objectives[name] for name in objectives_order}

    # def vectorize_constrains(self, constrains_order: List[Linear]) -> None:
    #     """vectorize_constrain [summary] make constrains a vectorize (in numpy) to accelerate speed.
    #     """
    #     assert self.variables_index is not None
    #     # collect objectives and check if objectives order is legal
    #     ordered_constrains: List[List[float]] = [[0.0] * len(self.variables) for _ in range(len(constrains_order))]
    #     for index, constrain in enumerate(constrains_order):
    #         for var, coef in constrain.coef.items():
    #             ordered_constrains[index][self.variables_index[var]] = coef
    #     self.constrains_matrix = Mat(constrains_order).T
    #     # set objectives attributes
    #     self.constrains_order = constrains_order
    #     self.constrains_num = len(constrains_order)
    #     self.objectives = {name: self.objectives[name] for name in constrains_order}

    def vectorize(self, objectives_order: List[str] = []) -> None:
        """vectorize [summary] vectorize the variables and objectives
        """
        self.vectorize_variables()
        if len(objectives_order) == 0:
            objectives_order = list(self.objectives.keys())
        self.vectorize_objectives(objectives_order)

    def vectorize_values(self, values: Dict[str, bool]) -> Mat:
        """vectorize_values [summary] make variables value (mapping variable name to bool value)
        a vector (one-line numpy matrix).

        Note that this function ignore variables not in self.variables.
        """
        vector: List[int] = [0] * len(self.variables)
        for var, val in values.items():
            if var not in self.variables_index: continue
            if not val: continue
            vector[self.variables_index[var]] = 1
        return Mat(vector)

    def evaluate_objectives(self, values: Dict[str, bool]) -> List[float]:
        """evaluate_objectives [summary] evaluate objectives list with variables values.
        """
        obj_values: List[float] = [0.0] * len(self.objectives_index)
        for obj_name, obj_index in self.objectives_index.items():
            for var, coef in self.objectives[obj_name].items():
                if values[var]:
                    obj_values[obj_index] += coef 
                    
        return obj_values
    
    def evaluate_offset_objectives(self, values: Dict[str, bool]) -> List[float]:
        """evaluate_objectives [summary] evaluate objectives list with variables values.
        """
        obj_values: List[float] = [0.0] * len(self.objectives_index)
        for obj_name, obj_index in self.objectives_index.items():
            for var, coef in self.offset_objectives[obj_name].items():
                if values[var]:
                    obj_values[obj_index] += coef 
                    
        return obj_values

    def evaluate_single_objective(self, values: Dict[str, bool], weights: Dict[str, float]) -> List[float]:
        """evaluate-single-objective [summary] evaluate single-objective
        with the sum of objectives list with variables values.
        """
        sum_obj = 0.0
        res: List[float] = []
        obj_values: List[float] = [0.0] * len(self.objectives_index)
        for obj_name, obj_index in self.objectives_index.items():
            for var, coef in self.objectives[obj_name].items():
                if values[var]:
                    obj_values[obj_index] += coef
                    sum_obj += obj_values[obj_index] * weights[obj_name]
        res.append(sum_obj)
        return res

    def evaluate_constraints(self, values: Dict[str, bool], violated_count: bool) -> int:
        """evaluate_constraints [summary] evaluate violated constriants count with variables values.
        The violated constraint would be count as a number when violated_count is True,
        otherwise it would only indicate feasible (0) or infeasible(1).
        """
        if violated_count:
            violated = 0
            for constraint in self.constraints:
                if not constraint.evaluate(values):
                    violated += 1
            return violated
        else:
            for constraint in self.constraints:
                if not constraint.evaluate(values):
                    return 1
            return 0
        
    def _evaluate(self, values: Dict[str, bool], offset_flag: bool = False, violated_count: bool = True) -> Tuple[List[float], int]:
        """_evaluate [summary] evaluate a solution with variables values.
        The violated constraint would be count as a number when violated_count is True,
        otherwise it would only indicate feasible (0) or infeasible(1).

        Return (objectives values, violated)
        """
        # evaluate objectives
        if offset_flag:
            obj_values = self.evaluate_offset_objectives(values)
        else:
            obj_values = self.evaluate_objectives(values)
        # evaluate violated
        violated = self.evaluate_constraints(values, violated_count)
        return obj_values, violated

    def _wso_evaluate(self, values: Dict[str, bool], weights: Dict[str, float], violated_count: bool = True) -> Tuple[List[float], int]:
        """_evaluate [summary] evaluate a solution with variables values.
        The violated constraint would be count as a number when violated_count is True,
        otherwise it would only indicate feasible (0) or infeasible(1).

        Return (objective values, violated)
        """
        # evaluate objectives
        obj_value = self.evaluate_single_objective(values, weights)
        # evaluate violated
        violated = self.evaluate_constraints(values, violated_count)
        return obj_value, violated

    def _empty_solution(self) -> BinarySolution:
        """_empty_solution [summary] prepare a empty BinarySolution.
        """
        # NOTE: we use one int BinarySolution.constraint to record violated constraints num.
        solution = BinarySolution(self.variables_num, self.objectives_num, self.constraints_num)
        solution.constraints[0] = 0
        return solution

    def listize_values(self, values: Dict[str, bool]) -> List[bool]:
        """listize_values [summary] make values dict a list based on variables indexing.
        """
        values_list: List[bool] = [False] * self.variables_num
        for index, var in enumerate(self.variables):
            if values[var]: values_list[index] = True
        return values_list

    def evaluate(self, values: Dict[str, bool]) -> BinarySolution:
        """evaluate [summary] evaluate a solution with variables values.
        Return a BinarySolution from jmetal.
        """
        # prepare a BinarySolution
        solution = self._empty_solution()
        # NOTE: note that jmetal use variables in this way, variables: [[True, False, True, ...]]
        solution.variables = [self.listize_values(values)]
        solution.objectives, solution.constraints[0] = self._evaluate(values, self.offset_flag, self.violateds_count)
        return solution

    def wso_evaluate(self, values: Dict[str, bool], weights: Dict[str, float]) -> BinarySolution:
        """evaluate [summary] evaluate a solution with variables values.
        Return a BinarySolution from jmetal.
        """
        # prepare a BinarySolution
        solution = self._empty_solution()
        # NOTE: note that jmetal use variables in this way, variables: [[True, False, True, ...]]
        solution.variables = [self.listize_values(values)]
        solution.objectives, solution.constraints[0] = self._wso_evaluate(values, weights, self.violateds_count)
        return solution

    def evaluate_solution(self, solution: BinarySolution, offset_flag: bool = False) -> BinarySolution:
        """evaluate_solution [summary] evaluate a given solution and return itself.
        """
        values = {var: solution.variables[0][ind] for ind, var in enumerate(self.variables)}
        if len(solution.constraints) == 0:
            solution.constraints.append([])
        solution.objectives, solution.constraints[0] = self._evaluate(values, offset_flag)
        return solution

    def weighted_objectives_sum(self, objectives_weight: Dict[str, float] = {}) -> Dict[str, float]:
        """weighted_objectives_sum [summary] get weighted sum objective.
        """
        objective: Dict[str, float] = {}
        if len(objectives_weight) == 0:
            objectives_weight = {k: 1 for k in self.objectives_order}
        for obj_name, obj_weight in objectives_weight.items():
            assert obj_name in self.objectives
            for var, coef in self.objectives[obj_name].items():
                if var in objective:
                    objective[var] += (obj_weight * coef)
                else:
                    objective[var] = (obj_weight * coef)
        return objective

    def convert_to_BinarySolution(self, bool_solution: List[bool]) -> Dict[str, bool]:
        bs: Dict[str, bool] = {}
        for index, sol in enumerate(bool_solution):
            bs[self.variables[index]] = sol
        return bs


class PymooProblem(Pro):
    """
    PymooProblem [summary] convert Problem formal to run in pymoo package.
    """
    def __init__(self, problem: Problem, **kwargs):
        self.n_var = problem.variables_num
        self.n_obj = problem.objectives_num
        # self.n_ieq_constr = problem.ieq_constraints_num
        # self.n_eq_constr = problem.eq_constraints_num
        self.problem = problem
        self.constraints_lp: List[Linear] = []
        self.vars = problem.variables
        self.n_eq_constr = 0
        self.n_ieq_constr = 0
        self.objs = []
        self.cons = []

        for constraint in self.problem.constraints:
            self.constraints_lp += constraint.to_linear()
        # self.problem.constraints = self.constraints_lp

        for constraint in self.constraints_lp:
            if constraint.sense == '=':
                self.n_eq_constr += 1
            else:
                self.n_ieq_constr += 1

        xl = [0 for _ in range(self.problem.variables_num)]
        xu = [1 for _ in range(self.problem.variables_num)]

        super().__init__(n_var=self.n_var, n_obj=self.n_obj, vars=self.vars,  n_ieq_constr=self.n_ieq_constr,
                         n_constr=len(self.constraints_lp), n_eq_constr=self.n_eq_constr, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Assuming the Algorithm has variability size N, the input variable x is a
        matrix with the [populationSize, N].
        """
        # objective
        objs = []
        for obj_name, obj_content in self.problem.objectives.items():
            num = []
            for k, v in obj_content.items():
                num.append(v)
            assert len(x[0]) == len(num), "objective's vars num is not equal x!"
            # auto transpose

            objs.append(np.dot(x, num))
        self.objs = objs

        # constrain
        ordered_constrains: List[List[float]] = [[0.0] * len(self.problem.variables) for _ in
                                                 range(len(self.constraints_lp))]
        for index, constrain in enumerate(self.constraints_lp):
            for var, coef in constrain.coef.items():
                assert var in self.problem.variables_index.keys()
                pos = self.problem.variables_index[var]
                ordered_constrains[index][pos] = coef
        cons = []
        for index, constrain in enumerate(self.constraints_lp):
            assert len(ordered_constrains[0]) == len(x[0]), "constraints' vars num is not equal x!"
            cons.append(np.dot(x, ordered_constrains[index]) - constrain.rhs)
        self.cons = cons

        out["F"] = objs
        out["G"] = cons

class PysooProblem(Pro):
    """
    PysooProblem [summary] convert Problem formal to run in pysoo package.
    """
    def __init__(self, problem: Problem, weights: Dict[str, float], **kwargs):
        self.n_var = problem.variables_num
        self.n_obj = problem.objectives_num
        # self.n_ieq_constr = problem.ieq_constraints_num
        # self.n_eq_constr = problem.eq_constraints_num
        self.problem = problem
        self.constraints_lp: List[Linear] = []
        self.vars = problem.variables
        self.n_eq_constr = 0
        self.n_ieq_constr = 0
        self.weights = weights
        self.objs = []
        self.cons = []

        for constraint in self.problem.constraints:
            self.constraints_lp += constraint.to_linear()
        # self.problem.constraints = self.constraints_lp

        for constraint in self.constraints_lp:
            if constraint.sense == '=':
                self.n_eq_constr += 1
            else:
                self.n_ieq_constr += 1

        xl = [0 for _ in range(self.problem.variables_num)]
        xu = [1 for _ in range(self.problem.variables_num)]

        super().__init__(n_var=self.n_var, n_obj=1, vars=self.vars,  n_ieq_constr=self.n_ieq_constr,
                         n_constr=len(self.constraints_lp), n_eq_constr=self.n_eq_constr, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Assuming the Algorithm has variability size N, the input variable x is a
        matrix with the [populationSize, N].
        """
        # objective
        objs = 0.0
        for obj_name, obj_content in self.problem.objectives.items():
            num = []
            for k, v in obj_content.items():
                num.append(v)
            assert len(x[0]) == len(num), "objective's vars num is not equal x!"
            # auto transpose

            objs += (np.dot(x, num) * self.weights[obj_name])
        self.objs = objs

        # constrain
        ordered_constrains: List[List[float]] = [[0.0] * len(self.problem.variables) for _ in
                                                 range(len(self.constraints_lp))]
        for index, constrain in enumerate(self.constraints_lp):
            for var, coef in constrain.coef.items():
                assert var in self.problem.variables_index.keys()
                pos = self.problem.variables_index[var]
                ordered_constrains[index][pos] = coef
        cons = []
        for index, constrain in enumerate(self.constraints_lp):
            assert len(ordered_constrains[0]) == len(x[0]), "constraints' vars num is not equal x!"
            cons.append(np.dot(x, ordered_constrains[index]) - constrain.rhs)
        self.cons = cons

        out["F"] = objs
        out["G"] = cons

class LP(Problem):
    """LP [summary] LP, short for Linear Problem.
    """
    TYPE = 'LP'

    def __init__(self, name: str, objectives_order: List[str] = []) -> None:
        super().__init__(name=name)
        # vectorize the problem
        self.vectorize(objectives_order)
        # convert all constraints in linear form
        self.constraints_lp: List[Linear] = []
        for constraint in self.constraints:
            self.constraints_lp += constraint.to_linear()

    def info(self) -> None:
        super().info()

    def get_objectives(self) -> List[Dict[str, float]]:
        return [self.objectives[obj_name] for obj_name in self.objectives_order]

    def get_constraints(self) -> List[Linear]:
        return self.constraints_lp


class QP(Problem):
    """QP [summary] QP, short for Quadratic Problem.
    """
    def __init__(self, name: str, objectives_order: List[str] = [], offset_flag: bool = False) -> None:
        super().__init__(name=name, offset_flag=offset_flag)
        # vectorize the problem
        self.vectorize(objectives_order)
        # convert all constrains in quadratic form
        self.artificial_list: List[str] = []
        self.constraints_qp: List[Quadratic] = []
        for constraint in self.constraints:
            self.constraints_qp.append(constraint.to_quadratic(self.artificial_list))
        self.constraint_sum: Quadratic = Constraint.quadratic_sum(self.constraints_qp)

        self.all_variables_index = {}
        assert len(self.variables_index) == self.variables_num
        for key, val in self.variables_index.items():
            self.all_variables_index[key] = val
        for index, var in enumerate(self.artificial_list):
            self.all_variables_index[var] = index + self.variables_num
        # assert len(self.all_variables_index) == len(self.variables) + len(self.artificial_list)
        

    def info(self) -> None:
        super().info()
        artificials_num = len(self.artificial_list)
        all_variables_num = len(self.variables) + len(self.artificial_list)
        print('all variables number: {} (artificial variables: {})'.format(all_variables_num, artificials_num))
