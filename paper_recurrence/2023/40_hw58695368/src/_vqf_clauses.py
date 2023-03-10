# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
"""Basic class of VQF-clauses."""

from abc import abstractmethod
from src.vqf_var_pool import VQFVarPool


class VQFClauses:  # pylint: disable=too-few-public-methods
    """
    Basic class for VQFClauses.

    Args:
        m_int (int): number to be factored, as an integer.
        p_q_int (list[int]): Two factors. Default: None.
    """

    def __init__(self, m_int, p_q_int, *args, **kwargs):
        """Initialize an VQFClauses object."""
        self._clauses = []
        self.v_p = VQFVarPool(m_int, p_q_int)
        self._implement(*args, **kwargs)

    @abstractmethod
    def _implement(self, *args, **kwargs):
        """Implement of clauses."""

    @property
    def clauses(self):
        """
        Get the clauses of this factorization problem.

        Returns:
            list[list[set, tuple]], the clauses of this factorization problem.
        """
        return self._clauses

    @property
    def n_clauses(self):
        """
        Get the number of clauses.

        Returns:
            int, the number of clauses.
        """
        return len(self._clauses)

    @property
    def all_variables(self):
        """
        Get all variables in clauses.

        Returns:
            list[str], variables in clauses.
        """
        return self.v_p.all_variables

    def get_ham(self, c_l, c_r):
        """
        Get hamiltonian according to clauses in the given range.

        Args:
            c_l (int): Left index of clauses.
            c_r (int): Right index of clauses.

        Returns:
            QubitOperator, the hamiltonian of clauses.
            list, variable corresponding to Pauli operation Z0~Zn.
        """
        var_l = set() # 区间范围子句中所有变量
        for c in self._clauses[c_l:c_r]:
            var_l |= c[0]
        self.v_p.build_var_oper_list(list(var_l)) # 建立所需变量与泡利算符的映射
        ham = 0 # 区间范围内所有子句哈密顿量平方和
        for c in self._clauses[c_l:c_r]:
            h_item = 0 # 单独一条子句的哈密顿量
            for c_item in c[1]:
                if isinstance(c_item, (list, tuple)): # 乘积项
                    h_i = 1
                    for i in c_item:
                        h_i *= self.v_p(i)
                else: # 独立项
                    h_i = self.v_p(c_item)
                h_item += h_i
            ham += h_item ** 2 # 平方后优化最小值0
        return ham, self.v_p.var_list

    def get_expr(self, index):
        """
        Generate expression based on clause.

        Args:
            index (int): The index of clause.
        """
        expr = ''
        vars = set()
        for c_item in self._clauses[index][1]:
            if isinstance(c_item, (list, tuple)): # 乘积项
                expr_i = ''
                for i in c_item:
                    e, v = self.v_p.get_expr(i)
                    vars |= set(v)
                    expr_i += f'*({e})'
                expr += expr_i[1:]
            else: # 独立项
                e, v = self.v_p.get_expr(c_item)
                vars |= set(v)
                expr += f'({e})'
            expr += '+'
        return expr[:-1], list(vars)

    def update_var(self, name, expr_form, value):
        """
        Update one variable.

        Args:
            name (str): Name of variable.
            expr_form (int): Expression form. Supported:
                {0: binary int, 1: x, 2: 1-x, 3: xy, 4: 1-xy}
            value (str): Variable name or a number.
        """
        self.v_p.update_var(name, expr_form, value)

    def get_result(self):
        """Get the result of VQF."""
        return self.v_p.get_result()
