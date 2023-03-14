# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
"""Variable pool for VQF."""

import re
from mindquantum.core.operators import QubitOperator


class VQFVar:
    """
    Args:
        name (str): Name of variable.
        expr_form (int): Expression form. Supported:
            {0: binary int, 1: x, 2: 1-x, 3: xy, 4: 1-xy}
        value (str): Variable name or a number.
    """

    def __init__(self, name, expr_form, value):
        """Initialize a VQFVar object."""
        self.id = name
        self._e_f = expr_form # expression form: {0: binary int, 1: x, 2: 1-x, 3: xy, 4: 1-xy}
        self._v = value

    def __call__(self, obj):
        """Get term."""
        if self._e_f == 0: # binary int
            return int(self._v)
        if self._e_f == 1: # x
            if self.id == self._v: # itself
                return obj.oper_list[obj.var_list.index(self._v)]
            else: # substitution
                return obj.v_dict[self._v](obj)
        if self._e_f == 2: # 1-x
            if self.id == self._v: raise
            return 1 - obj.v_dict[self._v](obj)
        if self._e_f in [3, 4]: # xy & 1-xy
            if self.id in self._v: raise
            v = obj.v_dict[self._v[0]](obj) * obj.v_dict[self._v[1]](obj)
            return v if self._e_f == 3 else 1 - v
        return None

    @property
    def is_const(self):
        """Check whether the variable is constant."""
        return False if self._e_f else True

    @property
    def is_ind_var(self):
        """Check whether the variable is independent variable."""
        if self._e_f == 1 and self.id == self._v:
            return True
        return False

    def get_expr(self, obj):
        """Get variable relation expression."""
        if self._e_f == 0:
            return str(self._v)
        if self._e_f == 1: # x
            if self.id == self._v: # itself
                return self._v
            else: # substitution
                return obj.v_dict[self._v].get_expr(obj)
        if self._e_f == 2: # 1-x
            if self.id == self._v: raise
            return f'(1-{obj.v_dict[self._v].get_expr(obj)})'
        if self._e_f in [3, 4]: # xy & 1-xy
            if self.id in self._v: raise
            l = obj.v_dict[self._v[0]].get_expr(obj)
            r = obj.v_dict[self._v[1]].get_expr(obj)
            if l == '0' or r == '0':
                v = '0'
            else:
                v = f'{l}*{r}'
            return v if self._e_f == 3 else f'(1-{v})'
        return None

    def clear(self, obj):
        """Convert a non-variable item to a constant value."""
        if self._e_f:
            try:
                v = eval(self.get_expr(obj))
                self._v = v
                self._e_f = 0
            except NameError:
                pass
            '''
            if not list(self.get_var(obj)):
                self._v = eval(self.get_expr(obj))
                self._e_f = 0
            '''

    def get_var(self, obj):
        """Get all independent variables."""
        if self.is_const:
            return set()
        if self.is_ind_var:
            return {self._v}
        return self._get_var(obj)

    def _get_var(self, obj):
        """Rules for searching all independent variables."""
        if self._e_f in [1, 2]: # x & 1-x
            return obj.v_dict[self._v].get_var(obj)
        if self._e_f in [3, 4]: # xy & 1-xy
            return obj.v_dict[self._v[0]].get_var(obj) | obj.v_dict[self._v[1]].get_var(obj)


class VQFVarPool:
    """
    Examples:
        >>> from src.vqf_var_pool import VQFVarPool, VQFVar
        >>> m, p, q = 56153, 241, 233
        >>> vp = VQFVarPool(m, [p, q])
        >>> vp.all_variables          # 查看所有变量(不包含被代换的变量)
        ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
        >>> vp.update_var('z_1_2', 1, 'z_1_2')  # 修改或增添变量(参数见`VQFVar`)
        >>> vp_e = vp.all_elements    # 查看所有元素(包含确定值和被代换的变量)
        >>> vp_e[:6]
        ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']
        >>> vp.build_var_oper_list(vp_e[:6])    # 构造变量与泡利算符的映射关系
        >>> vp.var_list   # 变量列表中的每个变量按顺序对应一个泡利算符
        ['p1', 'p2', 'p3', 'p4', 'p5']
        >>> vp('p2')      # 如 p2 对应 Z1
        0.5 [] +
        0.5 [Z1]
        >>> vp.update_var('p1', 1, 'p2')        # p1=p2
        >>> vp.update_var('p2', 2, 'p3')        # p2=1-p3
        >>> vp.update_var('p3', 1, 'p4')        # p3=p4
        >>> vp.build_var_oper_list(vp_e[:6])    # 构造变量与泡利算符的映射关系
        >>> vp.var_list   # 完成变量代换后，剩余变量分别对应 Z0、Z1 的泡利算符
        ['p4', 'p5']
        >>> [vp('p1'), vp('p2'), vp('p3'), vp('p4')] # 全部换成了 p4 对应的 Z0
        [0.5 [] +
         -0.5 [Z0] ,
         0.5 [] +
         -0.5 [Z0] ,
         0.5 [] +
         0.5 [Z0] ,
         0.5 [] +
         0.5 [Z0] ]
        >>> vp.update_var('p1', 3, ('p2','p3'))      # p1=p2p3
        >>> vp.update_var('p2', 2, 'p4')             # p2=1-p4
        >>> vp.update_var('p3', 2, 'p5')             # p3=1-p5
        >>> vp.get_expr('p1')                        # 获得变量表达式形式
        ('(1-p4)*(1-p5)', ['p5', 'p4'])
        >>> vp.update_var('p1', 1, 'p0')             # p1=p2
        >>> vp.update_var('p2', 3, ('p4', 'p5'))     # p2=p4*p5
        >>> vp.update_var('p3', 4, ('p4', 'p5'))     # p3=1-p4*p5
        >>> vp.update_var('p4', 1, 'q3')             # p4=q3
        >>> vp.build_var_oper_list(vp_e[:6])    # 虽然构造操作算符的变量只选取了q0~q5
        >>> vp.var_list                         # 还是能根据关系式搜索到自变量 q3
        ['q3', 'p5']
        >>> [vp('p1'), vp('p2'), vp('p3'), vp('p4')]
        [1,
         0.25 [] +
         0.25 [Z0] +
         0.25 [Z0 Z1] +
         0.25 [Z1] ,
         0.75 [] +
         -0.25 [Z0] +
         -0.25 [Z0 Z1] +
         -0.25 [Z1] ,
         0.5 [] +
         0.5 [Z0] ]
        >>> vp('p4') * vp('p5')                 # 可以验证一下 p2=p4*p5
        0.25 [] +
        0.25 [Z0] +
        0.25 [Z0 Z1] +
        0.25 [Z1]
        >>> _ = [vp.update_var(f'p{i}', 2, 'p0') for i in range(1, 5)]
        >>> vp.get_expr('p4')
        ('(1-1)', [])
        >>> vp.clear()                          # 常数式散去变量
        >>> [vp(f'p{i}') for i in range(5)]
        [1, 0, 0, 0, 0]
        >>> _ = [vp.update_var(k, 1, 'p0') for k in vp.v_dict.keys()]
        >>> vp.update_var('p0', 0, 1)
        >>> vp.update_var('q0', 0, 1)
        >>> vp.get_result()                     # 获取结果
        {'status': True, 'p': 65535, 'q': 255, 'm': 56153, 'p_q_real': None}
    """

    def __init__(self, m_int, p_q_int):
        """Initialize a VQFVarPool object."""
        self.m_dict = dict() # dictionary of binary m
        self.v_dict = dict() # dictionary for p, q or other variables
        self.n_m = 0         # bits of binary m
        self.n_p = 0         # bits of binary p
        self.n_q = 0         # bits of binary q
        self._build_init_dict(m_int, p_q_int)
        self.var_list = []   # variable list
        self.oper_list = []  # Pauli Operator list, corresponding to variable list
        self._m_int = m_int
        self._p_q_int = p_q_int

    def __call__(self, key):
        """Get value with key."""
        try:
            return self.v_dict[key](self)
        except KeyError:
            return key

    @property
    def all_elements(self):
        """All elements in v_dict, including constant items and related items."""
        return sorted(list(self.v_dict.keys()),
                      key=lambda l:(l[0], *[int(g.group()) for g in re.finditer('\d+', l)]))

    @property
    def all_variables(self):
        """All variables in v_dict."""
        v_list = set()
        for i in self.v_dict.values():
            v_list |= i.get_var(self)
        return sorted(list(v_list),
                      key=lambda l:(l[0], *[int(g.group()) for g in re.finditer('\d+', l)]))

    def update_var(self, name, expr_form, value):
        """Update one variable."""
        self.v_dict[name] = VQFVar(name, expr_form, value)
        self.v_dict[name].clear(self)

    def clear(self):
        """Convert each non-variable item to constant value."""
        [var.clear(self) for var in self.v_dict.values()]

    def get_expr(self, key):
        """Get variable relation expression."""
        try:
            return self.v_dict[key].get_expr(self), list(self.v_dict[key].get_var(self))
        except KeyError:
            return str(key), list()

    def get_result(self):
        """Get the result of VQF."""
        self.clear()
        p = self._get_result('p', self.n_p-1)
        q = self._get_result('q', self.n_q-1)
        if p is None and q is None:
            return {'status': False}
        return {'status': True,
                'p': p,
                'q': q,
                'm': self._m_int,
                'p_q_real': self._p_q_int}

    def build_var_oper_list(self, var_key_list):
        """Build variable list and operator list."""
        self._build_var_list(var_key_list)
        self.oper_list = []
        for i in range(len(self.var_list)):
            self.oper_list.append((QubitOperator(f'Z{i}')+1)/2)

    def _build_var_list(self, var_key_list):
        """Build variable list."""
        self.var_list = set()
        for k in var_key_list:
            self.var_list |= self.v_dict[k].get_var(self)
        self.var_list = list(self.var_list)
        self.var_list.sort(key=lambda l:
                           tuple([int(g.group()) for g in re.finditer('\d+', l)]))

    def _build_init_dict(self, m_int, p_q_int):
        """Build dictionaries representing m, p and q."""
        m_bin = bin(m_int)[2:][::-1]
        for i, item in enumerate(m_bin):
            self.m_dict[i] = int(item)
        self.n_m = i + 1
        self.v_dict['p0'] = VQFVar('p0', 0, 1)
        self.v_dict['q0'] = VQFVar('q0', 0, 1)
        if p_q_int is None:
            self.n_p = self.n_m
            self.n_q = (self.n_p + 1) // 2 # p＞q
        else:
            p_bin = bin(p_q_int[0])[2:][::-1]
            q_bin = bin(p_q_int[1])[2:][::-1]
            self.n_p = len(p_bin)
            self.n_q = len(q_bin)
        for i in range(1, self.n_p):
            self.update_var(f'p{i}', 1, f'p{i}')
        for i in range(1, self.n_q):
            self.update_var(f'q{i}', 1, f'q{i}')
        if p_q_int is not None:
            self.update_var(f'p{self.n_p-1}', 0, 1)
            self.update_var(f'q{i}', 0, 1)

    def _get_result(self, l, n):
        """Get the result of VQF."""
        _s = 0
        for i in range(n, -1, -1):
            _s <<= 1
            _v = self.v_dict[f'{l}{i}']
            if not _v.is_const:
                return None
            _s += _v(self)
        return _s
