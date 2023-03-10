# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
"""Clauses library for various VQFs."""

from itertools import product
from src._vqf_clauses import VQFClauses


class VQFClausesA(VQFClauses):
    """
    [1] `Variational Quantum Factoring`
        (https://arxiv.org/abs/1808.08927)
    [2] https://github.com/mstechly/vqf

    Args:
        m_int (int): number to be factored, as an integer.
        p_q_int (list[int]): Two factors. Default: None.

    Examples:
        >>> from src.vqf_clauses_lib import VQFClausesA
        >>> m, p, q = 56153, 241, 233
        >>> vc = VQFClausesA(m, [p, q])
        >>> vc.clauses[:3]      # 看一下前三句子句
        [[{'p0', 'q0'}, (['p0', 'q0'], -1)],
         [{'p0', 'p1', 'q0', 'q1', 'z_1_2'},
          (['p0', 'q1'], ['p1', 'q0'], 0, [-2, 'z_1_2'])],
         [{'p0', 'p1', 'p2', 'q0', 'q1', 'q2', 'z_1_2', 'z_2_3', 'z_2_4'},
          (['p0', 'q2'],
           ['p1', 'q1'],
           ['p2', 'q0'],
           'z_1_2',
           0,
           [-2, 'z_2_3'],
           [-4, 'z_2_4'])]]
        >>> for i in range(2):   # 看一下前两句子句的表达式形式
                print(vc.get_expr(i))
        ('(1)*(1)+(-1)', [])
        ('(1)*(q1)+(p1)*(1)+(0)+(-2)*(z_1_2)', ['p1', 'q1', 'z_1_2'])
        >>> vc.get_ham(0, 1)[0]  # 第一句子句 p0*q0-1 由于 p0、q0 初始值都为1因此等于0
        0
        >>> vc.get_ham(1, 3)[0]  # 第二和第三句子句结合在一起的哈密顿量
        9.0 [] +
        -0.5 [Z0] +
        0.25 [Z0 Z1 Z2] +
        0.25 [Z0 Z1 Z3] +
        0.25 [Z0 Z1 Z4] +
        -0.5 [Z0 Z1 Z5] +
        -1.0 [Z0 Z1 Z6] +
        -0.75 [Z0 Z2] +
        0.25 [Z0 Z3] +
        0.25 [Z0 Z4] +
        -0.5 [Z0 Z5] +
        -1.0 [Z0 Z6] +
        -0.5 [Z1] +
        -0.75 [Z1 Z2] +
        0.25 [Z1 Z3] +
        0.25 [Z1 Z4] +
        -0.5 [Z1 Z5] +
        -1.0 [Z1 Z6] +
        -1.25 [Z2] +
        0.5 [Z2 Z3] +
        0.5 [Z2 Z4] +
        -1.0 [Z2 Z5] +
        -2.0 [Z2 Z6] +
        -1.25 [Z3] +
        0.5 [Z3 Z4] +
        -1.0 [Z3 Z5] +
        -2.0 [Z3 Z6] +
        -1.25 [Z4] +
        -1.0 [Z4 Z5] +
        -2.0 [Z4 Z6] +
        2.5 [Z5] +
        4.0 [Z5 Z6] +
        5.0 [Z6]
        >>> vc.v_p.var_list  # 第二和第三句子句所包含的变量，对应 Z0~Z6
        ['p1', 'q1', 'z_1_2', 'q2', 'p2', 'z_2_3', 'z_2_4']
        >>> vc.update_var('p1', 1, 'q1')     # 更新关系式 p1=q1
        >>> vc.update_var('z_1_2', 0, 0)     # 更新关系式 z_1_2=0
        >>> vc.update_var('p2', 1, 'q2')     # 更新关系式 p2=q2
        >>> vc.get_ham(1, 3)[0]              # 再看一下二三子句
        10.5 [] +
        0.5 [Z0] +
        1.0 [Z0 Z1] +
        -1.0 [Z0 Z2] +
        -2.0 [Z0 Z3] +
        -3.0 [Z1] +
        -2.0 [Z1 Z2] +
        -4.0 [Z1 Z3] +
        3.0 [Z2] +
        4.0 [Z2 Z3] +
        6.0 [Z3]
        >>> vc.v_p.var_list                  # 有效减少变量数量
        ['q1', 'q2', 'z_2_3', 'z_2_4']
    """

    def __init__(self, m_int, p_q_int):
        """Initialize a VQFClausesA object."""
        super().__init__(m_int, p_q_int)

    def _implement(self):
        """Implement of clauses."""
        n_c = self.v_p.n_p + self.v_p.n_q
        clauses_l = [[] for _ in range(n_c)]
        clauses_r = [[] for _ in range(n_c)]
        clauses_v = [set() for _ in range(n_c)]
        for i, j in product(range(self.v_p.n_p), range(self.v_p.n_q)):
            clauses_l[i+j].append([f'p{i}', f'q{j}'])
            clauses_v[i+j] |= {f'p{i}', f'q{j}'}
        for i in range(n_c):
            clauses_r[i].append(-self.v_p.m_dict.get(i, 0))
            carry = len(bin(len(clauses_l[i]))) - 2
            for c in range(1, carry):
                if i+c >= n_c:
                    break
                self.v_p.update_var(f'z_{i}_{i+c}', 1, f'z_{i}_{i+c}')
                clauses_r[i].append([-1<<c, f'z_{i}_{i+c}'])
                clauses_v[i] |= {f'z_{i}_{i+c}'}
                clauses_l[i+c].append(f'z_{i}_{i+c}')
                clauses_v[i+c] |= {f'z_{i}_{i+c}'}
        self._clauses = list(map(lambda x, y, z:[z, (*x, *y)], clauses_l, clauses_r, clauses_v))


class VQFClausesB(VQFClauses):
    """
    [1] `The role of symmetries in adiabatic quantum algorithms`
        (https://arxiv.org/abs/0708.1882)

    Args:
        m_int (int): number to be factored, as an integer.
        p_q_int (list[int]): Two factors. Default: None.
    """

    def __init__(self, m_int, p_q_int):
        """Initialize a VQFClausesB object."""
        super().__init__(m_int, p_q_int)

    def _implement(self):
        """Implement of clauses."""
        pass
