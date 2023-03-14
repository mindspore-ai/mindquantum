# -*- coding: utf-8 -*-
"""
@NE
"""

class VQFPreprocessor:
    """
    """

    def __init__(self):
        self.name = 'VQFPreprocessor'

    def algebraic_variable_elimination(self, vc):
        """
        Simplify clauses with Algebraic Variable Elimination.
        [1] `Factoring as Optimization`
            (https://www.microsoft.com/en-us/research/publication/factoring-as-optimization/)
        [2] `Variational Quantum Factoring`
            (https://arxiv.org/abs/1808.08927)
        [3] https://github.com/mstechly/vqf
        """
        pass

    def sliding_qaoa(self, vc, n, th):
        """
        Simplify clauses with sliding QAOA.
        假设与子句c相邻的n句子句的变量关联度大于n以外的非相邻子句，
        采用QAOA对该2n+1句子句进行优化，
        对最终态进行(Z_i、Z_iZ_j)测量，
        对概率大于th的结果可以有(0、1、y=x、y=1-x)等确定值或关系式。
        """
        pass

    def cheat(self, vc):
        """
        Factorization of number 143/56153/291311.
        [1] `Quantum factorization of 56153 with only 4 qubits`
            (https://arxiv.org/abs/1411.6758)
        """
        if vc.v_p._m_int not in [143, 56153, 291311]:
            return
        {143:self._cheat_143,
         56153:self._cheat_56153,
         291311:self._cheat_291311}.get(vc.v_p._m_int, None)(vc)
    def _cheat_143(self, vc):
        pass
    def _cheat_56153(self, vc):
        vc._clauses = [[{'p3', 'q3'}, ('p3', 'q3', -1)],
            [{'p4', 'q4'}, ('p4', 'q4', -1)],
            [{'p3', 'p4', 'q3', 'q4'}, (['p4', 'q3'], ['p3', 'q4'], -1)]]
        p_dict = {0: 1, 1: 0, 2: 0, 3: 'p3', 4: 'p4', 5: 1, 6: 1, 7: 1}
        q_dict = {0: 1, 1: 0, 2: 0, 3: 'q3', 4: 'q4', 5: 1, 6: 1, 7: 1}
        self._cheat_build_p_q_dict(vc, p_dict, q_dict)
    def _cheat_291311(self, vc):
        vc._clauses = [[{'p1', 'q1'}, ('p1', 'q1', -1)],
            [{'p2', 'q2'}, ('p2', 'q2', -1)],
            [{'p5', 'q5'}, ('p5', 'q5', -1)],
            [{'p1', 'p2', 'q1', 'q2'}, (['p1', 'q2'], ['p2', 'q1'], -1)],
            [{'p2', 'p5', 'q2', 'q5'}, (['p2', 'q5'], ['p5', 'q2'], 0)],
            [{'p1', 'p5', 'q1', 'q5'}, (['p1', 'q5'], ['p5', 'q1'], -1)]]
        p_dict = {0: 1, 1: 'p1', 2: 'p2', 3: 1, 4: 0, 5: 'p5', 6: 0, 7: 0, 8: 0, 9: 1}
        q_dict = {0: 1, 1: 'q1', 2: 'q2', 3: 1, 4: 0, 5: 'q5', 6: 0, 7: 0, 8: 0, 9: 1}
        self._cheat_build_p_q_dict(vc, p_dict, q_dict)
    def _cheat_build_p_q_dict(self, vc, p_d, q_d):
        _ = [vc.update_var(k, 0, 0) for k in vc.v_p.v_dict.keys()]
        self._cheat_build_dict(vc, 'p', p_d)
        self._cheat_build_dict(vc, 'q', q_d)
    def _cheat_build_dict(self, vc, l, d):
        for k in d.keys():
            if isinstance(d[k], int):
                vc.update_var(f'{l}{k}', 0, d[k])
            else:
                vc.update_var(f'{l}{k}', 1, d[k])
