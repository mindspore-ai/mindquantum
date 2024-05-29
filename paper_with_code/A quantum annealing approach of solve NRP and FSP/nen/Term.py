from typing import List, Dict, Tuple, Any
import copy


class Linear:
    """ [summary] Binary Linear Constraint, polynomial sense rhs
        such as {x1: 1, x2: 2}, <=, 5 denotes x1 + 2 x2 <= 5
        sense should be <= (less-or-equal) or = (equal)
    """
    def __init__(self, coef={}, sense='<=', rhs=None) -> None:
        self.coef = coef
        self.sense = sense
        self.rhs = rhs

    def __str__(self):
        if not self.coef:
            return '<empty linear constraint>'
        s = ''
        first = True
        for k, v in self.coef.items():
            if first: first = False
            else: s += ' + '
            s += ('{}: {}'.format(k, v))
        s += ' '
        s += self.sense
        s += ' '
        s += str(self.rhs)
        return s


class Quadratic:
    """ [summary] Binary Quadratic Polynomial, composed by quadratic terms, linear terms and constant.
    For example, quadratic {(x1, x2): 2}, linear {x1: 1, x2: 3}, constant 5 denotes x1 + 3 x2 + 2 x1x2 + 5.
    Notes that, the square of a binary variable equals to itself, x1 * x1 == x1,
    thus pairs such as (x1, x1) is illegal.
    """
    def __init__(self, quadratic={}, linear={}, constant=0.0) -> None:
        self.quadratic = quadratic
        self.linear = linear
        self.constant = constant

    def empty(self) -> bool:
        return not (self.quadratic or self.linear)

    def __str__(self) -> str:
        if not self.quadratic and not self.linear: return '<empty quadratic constraint>'
        s = ''
        first = True
        for (k1, k2), v in self.quadratic.items():
            if first: first = False
            else: s += ' + '
            if k1 > k2: k1, k2 = k2, k1
            s += ('({}, {}): {}').format(k1, k2, v)
        for k, v in self.linear.items():
            if first: first = False
            else: s += ' + '
            s += ('{}: {}').format(k, v)
        if self.constant != 0:
            if first: first = False
            else: s += ' + '
            s += ('constant: {}').format(self.constant)
        return s


class Constraint:
    """ [summary] fi or f is subfeature, f' is the parent feature
        left sense right
        used:
            linear inequation: {fi: ci} <= constant
            linear equation: {fi: ci} = constant
            mandatory: f <=> f'
            optional: f => f'
            exclude: f >< f'
            or subfeatures: [fi] or f'
            alternative subfeatures: [fi] alt f'

        Note that, all coefficients in Constriant should be INTEGER.
    """
    def __init__(self, left=None, sense=None, right=None) -> None:
        self.left = left
        self.sense = sense
        self.right = right
        # check if is legal
        self.check()

    def __str__(self) -> str:
        return str(self.left) + ' ' + self.sense + ' ' + str(self.right)

    def to_dict(self) -> List[Any]:
        return [self.left, self.sense, self.right]

    def check(self) -> None:
        """check [summary] check if constraint is legal
        """
        if self.left is None and self.sense is None and self.right is None:
            return
        elif self.sense == '<=' or self.sense == '=':
            assert isinstance(self.left, dict)
            # TODO: JUST REMOVE THIS ASSERT FOR NRP EXPERIMENTS
            # assert isinstance(self.right, int), (self.right, type(self.right))
        elif self.sense == '<=>' or self.sense == '=>' or self.sense == '><':
            assert isinstance(self.left, str)
            assert isinstance(self.right, str)
        elif self.sense == 'or' or self.sense == 'alt':
            assert isinstance(self.left, list)
            assert isinstance(self.right, str)
        else:
            assert False, 'illegal sense {}'.format(self.sense)

    def to_linear(self) -> List[Linear]:
        """to_linear [summary] convert a constraint to some linear constraints.
        """
        coef = {}
        sense = None
        rhs = None
        if self.sense == '=' or self.sense == '<=':
            # linear
            coef = self.left
            sense = self.sense
            rhs = self.right
            return [Linear(coef, sense, rhs)]
        elif self.sense == '<=>':
            # iff.
            coef[self.left] = 1
            coef[self.right] = -1
            sense = '='
            rhs = 0
            return [Linear(coef, sense, rhs)]
        elif self.sense == '=>':
            # dependency, left depends on right
            coef[self.left] = 1
            coef[self.right] = -1
            sense = '<='
            rhs = 0
            return [Linear(coef, sense, rhs)]
        elif self.sense == 'or':
            # or subfeatures
            linears = []
            # t1 <= f and t2 <= f and ...
            for left in self.left:
                linears.append(Linear({left: 1, self.right: -1}, '<=', 0))
            coef = {}
            # t1 + t2 + ... >= f
            for left in self.left:
                coef[left] = -1
            coef[self.right] = 1
            linears.append(Linear(coef, '<=', 0))
            return linears
        elif self.sense == '><':
            # exclude
            coef[self.left] = 1
            coef[self.right] = 1
            sense = '<='
            rhs = 1
            return [Linear(coef, sense, rhs)]
        elif self.sense == 'alt':
            # alt subfeatures
            linears = []
            # t1 <= f and t2 <= f and ...
            for left in self.left:
                linears.append(Linear({left: 1, self.right: -1}, '<=', 0))
            # t1 + t2 + ... >= f
            for left in self.left:
                coef[left] = 1
            coef[self.right] = -1
            linears.append(Linear(coef, '<=', 0))
            # t1 + t2 + ... <= 1
            for left in self.left:
                coef[left] = 1
            linears.append(Linear(coef, '<=', 1))
            return linears
        assert False

    @staticmethod
    def linear_poly_square(poly: Dict[str, int], constant: int) -> Quadratic:
        """linear_poly_square [summary] get the square of a linear polymonial.
        """
        linear: Dict[str, int] = {}
        quadratic: Dict[Tuple[str, str], int] = {}
        keys = poly.keys()
        # quadratic terms, note that xi * xi is same to xi, thus it is count as xi (linear)
        for k1 in keys:
            v1 = poly[k1]
            for k2 in keys:
                if k1 == k2:
                    linear[k1] = v1 ** 2
                else:
                    if (k1, k2) in quadratic:
                        quadratic[(k1, k2)] += v1 * poly[k2]
                    elif (k2, k1) in quadratic:
                        quadratic[(k2, k1)] += v1 * poly[k2]
                    else:
                        quadratic[(k1, k2)] = v1 * poly[k2]
        # linear terms
        for k in keys:
            linear[k] += (2 * constant * poly[k])
        return Quadratic(quadratic, linear, constant * constant)

    @staticmethod
    def poly_range(poly: Dict[str, int]) -> Tuple[int, int]:
        """poly_range [summary] lower and upper bound of a linear polynomial.
        """
        low, up = 0, 0
        for v in poly.values():
            if v > 0: up += v
            elif v < 0: low += v
        return low, up

    @staticmethod
    def linear_equation_quadratic(poly: Dict[str, int], rhs: int) -> Quadratic:
        """linear_equation_quadratic [summary] linear equation (constraint, =) to quadratic polynomial (minimize).
        """
        return Constraint.linear_poly_square(poly, -rhs)

    @staticmethod
    def linear_inequation_quadratic(poly: Dict[str, int], rhs: int, artificial_list: List[str]) -> Quadratic:
        """linear_inequation_quadratic [summary] linear inequation (constraint, <=) to quadratic polynomial (minimize).
        """
        # range of polynomial
        lb, _ = Constraint.poly_range(poly)
        # range of the slacken variable
        assert lb <= rhs
        value_range = rhs - lb
        artificial_variable_num = value_range.bit_length()
        artificial_id = len(artificial_list)
        base2 = 1
        # coefs for each slacken variables are
        # 1, 2, 4, ..., floor(log2(r)-0.1), r - floor(log2(r-0.1))
        for i in range(artificial_variable_num - 1):
            var_name = '${}'.format(artificial_id + i)
            artificial_list.append(var_name)
            poly[var_name] = base2
            base2 <<= 1
        var_name = '${}'.format(artificial_id + artificial_variable_num - 1)
        artificial_list.append(var_name)
        poly[var_name] = value_range - base2 + 1
        return Constraint.linear_equation_quadratic(poly, rhs)

    @staticmethod
    def quadratic_sum(quadratic_list: List[Quadratic]) -> Quadratic:
        """quadratic_sum [summary] get the sum of all quadratic polynomial.
        """
        quadratic: Dict[Tuple[str, str], int] = {}
        linear: Dict[str, int] = {}
        constant = 0
        for quadratic_constraint in quadratic_list:
            for (k1, k2), v in quadratic_constraint.quadratic.items():
                if (k1, k2) in quadratic:
                    quadratic[(k1, k2)] += v
                elif (k2, k1) in quadratic:
                    quadratic[(k2, k1)] += v
                else:
                    quadratic[(k1, k2)] = v
            for k, v in quadratic_constraint.linear.items():
                if k in linear:
                    linear[k] += v
                else:
                    linear[k] = v
            constant += quadratic_constraint.constant
        # reduce 0 value item
        quadratic = {k: v for k, v in quadratic.items() if v != 0}
        linear = {k: v for k, v in linear.items() if v != 0}
        return Quadratic(quadratic, linear, constant)

    def to_quadratic(self, artificial_list: List[str]) -> Quadratic:
        """to_quadratic [summary] convert constraint to quadratic polynomial (minimize).

        Args:
            artificial_list (List[str]): [description] artificial list,
            record new variables introduced during convertion.
        """
        if self.sense == '=':
            # linear equation
            return Constraint.linear_equation_quadratic(self.left.copy(), self.right)
        elif self.sense == '<=':
            # linear inequation '<=', slacken variables added
            return Constraint.linear_inequation_quadratic(self.left.copy(), self.right, artificial_list)
        elif self.sense == '<=>':
            # iff. x <=> y is same as x - y = 0
            return Constraint.linear_equation_quadratic({self.left: 1, self.right: -1}, 0)
        elif self.sense == '=>':
            # dependency, left depends on right
            # x => y <=> x <= y <=> (x - xy)^2 <=> x - xy
            # return Constraint.linear_inequation_quadratic({self.left: 1, self.right: -1}, 0, artificial_list)
            return Quadratic({(self.left, self.right): -1}, {self.left: 1}, 0)
        elif self.sense == 'or':
            # or subfeatures
            quadratic_list: List[Quadratic] = []
            # for each left_var => self.right
            for left_var in self.left:
                # dependency = Constraint.linear_inequation_quadratic({left_var: 1, self.right: -1}, 0, artificial_list)
                dependency = Quadratic({(left_var, self.right): -1}, {left_var: 1}, 0)
                quadratic_list.append(dependency)
            # sum{left_var} >= right <=> sum{-left_var} + right <= 0
            poly = {k: -1 for k in self.left}
            poly[self.right] = 1
            sum_constraint = Constraint.linear_inequation_quadratic(poly, 0, artificial_list)
            quadratic_list.append(sum_constraint)
            return Constraint.quadratic_sum(quadratic_list)
        elif self.sense == '><':
            # exclude, x >< y <=> xy
            return Quadratic({(self.left, self.right): 1}, {}, 0)
        elif self.sense == 'alt':
            # alt subfeatures
            alt_list: List[Quadratic] = []
            # for each left_var => self.right
            for left_var in self.left:
                # dependency = Constraint.linear_inequation_quadratic({left_var: 1, self.right: -1}, 0, artificial_list)
                dependency = Quadratic({(left_var, self.right): -1}, {left_var: 1}, 0)
                alt_list.append(dependency)
            # sum{left_var} >= right <=> sum{-left_var} + right <= 0
            poly = {k: -1 for k in self.left}
            poly[self.right] = 1
            sum_constraint = Constraint.linear_inequation_quadratic(poly, 0, artificial_list)
            alt_list.append(sum_constraint)
            # sum{left_var} <= 1
            poly = {k: 1 for k in self.left}
            sum_constraint = Constraint.linear_inequation_quadratic(poly, 1, artificial_list)
            alt_list.append(sum_constraint)
            return Constraint.quadratic_sum(alt_list)
        assert False

    # def evaluate(self, values: Dict[str, bool]) -> bool:
    #     """evaluate [summary] return if this constraint is violated
    #     """
    #     if self.sense == '<=' or self.sense == '=':
    #         left_value = 0
    #         for var, coef in self.left.items():
    #             if values[var]: left_value += coef
    #         if self.sense == '=': return (left_value == self.right)
    #         else: return left_value <= self.right
    #     elif self.sense == 'or' or self.sense == 'alt':
    #         # for each left < right
    #         if not values[self.right]:
    #             for left_var in self.left:
    #                 if values[left_var]: return False
    #         # sum of left >= right
    #         right_value = 1 if values[self.right] else 0
    #         left_value = 0
    #         for left_var in self.left:
    #             if values[left_var]: left_value += 1
    #         if left_value > right_value: return False
    #         # for alt, sum of left should not be greater than 1
    #         if self.sense == 'alt' and left_value > 1: return False
    #         return True
    #     elif self.sense == '<=>':
    #         return values[self.left] == values[self.right]
    #     elif self.sense == '=>':
    #         return not (values[self.left] and (not values[self.right]))
    #     elif self.sense == '><':
    #         return not (values[self.left] and values[self.right])
    #     assert False, ('cannot evaluate with this constraint', self.__str__())

    def evaluate(self, values: Dict[str, bool]) -> bool:
        """evaluate [summary] return if this constraint is violated
        """
        precision = 1e-9
        if self.sense == '<=' or self.sense == '=':
            left_value = 0
            for var, coef in self.left.items():
                if values[var]: left_value += coef
            if self.sense == '=': return ((left_value <= (self.right + precision)) and (left_value >= (self.right - precision)))
            else: return left_value <= (self.right + precision)
        elif self.sense == 'or':
            if values[self.right]:
                for left_var in self.left:
                    if values[left_var]: return True
                return False
            else:
                for left_var in self.left:
                    if values[left_var]: return False
                return True
        elif self.sense == 'alt':
            num: int = 0
            if values[self.right]:
                for left_var in self.left:
                    if values[left_var]: 
                        num += 1
                        if num > 1:
                            return False
                return num == 1
            else:
                for left_var in self.left:
                    if values[left_var]: return False
                return True
        elif self.sense == '<=>':
            return values[self.left] == values[self.right]
        elif self.sense == '=>':
            return not (values[self.left] and (not values[self.right]))
        elif self.sense == '><':
            return not (values[self.left] and values[self.right])
        assert False, ('cannot evaluate with this constraint', self.__str__())

    @staticmethod
    def quadratic_weighted_add(c1: float, c2: float, q1: Quadratic, q2: Quadratic) -> Quadratic:
        q = Quadratic({}, {}, c1 * q1.constant + c2 * q2.constant)
        for k, v in q1.quadratic.items():
            q.quadratic[k] = c1 * v
        for (k1, k2), v in q2.quadratic.items():
            if (k1, k2) in q.quadratic:
                q.quadratic[(k1, k2)] += (c2 * v)
            elif (k2, k1) in q.quadratic:
                q.quadratic[(k2, k1)] += (c2 * v)
            else:
                q.quadratic[(k1, k2)] = (c2 * v)
        for k, v in q1.linear.items():
            q.linear[k] = c1 * v
        for k, v in q2.linear.items():
            if k in q.linear:
                q.linear[k] += (c2 * v)
            else:
                q.linear[k] = (c2 * v)
        return q

    @staticmethod
    def quadratic_to_qubo_dict(quadratic: Quadratic) -> Dict[Tuple[str, str], float]:
        """quadratic_to_qubo_dict [summary] convert Quadratic(quadratic, linear, constant) to a qubo dict,
        linear key would be (k, k) as the same format with quadratic keys (k1, k2), constant would be dropped.
        """
        qubo: Dict[Tuple[str, str], float] = copy.copy(quadratic.quadratic)
        for k, v in quadratic.linear.items():
            qubo[(k, k)] = v
        return qubo
