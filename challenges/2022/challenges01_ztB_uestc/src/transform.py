# -*- coding: utf-8 -*-
#   Copyright (c) 2020 Huawei Technologies Co.,ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# from math import floor, log
# import numpy as np

# from mindquantum.core.operators.fermion_operator import FermionOperator
from mindquantum.core.operators.qubit_operator import QubitOperator
from multiprocessing import Pool, cpu_count


# 在源码的基础上进行了一点简单的优化，并用了多进程来加快哈密顿量的转化
# 源码地址：https://www.mindspore.cn/mindquantum/docs/zh-CN/r0.7/_modules/mindquantum/algorithm/nisq/chem/transform.html#Transform

def jordan_wigner(operator):
    terms = operator.terms
    length = len (terms)
    
    if length >= 100:
        cores = 2
        length //= cores
        pool = Pool(cores)
        data_list = []
        for i in range(cores):
            if i != cores - 1:
                data_list.append((dict(list(terms.items())[i*length:(i+1)*length]), operator))
            else:
                data_list.append((dict(list(terms.items())[i*length:]), operator))
        res = pool.map(process, data_list)
        pool.close()
        pool.join()

        return sum(res)
    else:
        return process((terms, operator))


def process(data_in): 
    transf_op = QubitOperator()
    y1 = []
    # coefficient_1 = 0.5
    
    for term in data_in[0]:
        # Initialize identity matrix.
        transformed_term = QubitOperator((), data_in[1].terms[term])
        
        # Loop through operators, transform and multiply.
        for ladder_operator in term:

            # Define lists of qubits to apply Pauli gates
            index = ladder_operator[0]
            x1 = [index]
            z1 = list(range(index))
            
            coefficient_2 = -.5j if ladder_operator[1] else .5j
            t_z = tuple((index, 'Z') for index in z1)
            transf_op_1 = QubitOperator(
                tuple((index, 'X') for index in x1) + \
                tuple((index, 'Y') for index in y1) + \
                t_z, 0.5)
            transf_op_2 = QubitOperator(
                tuple((index, 'X') for index in y1) + \
                tuple((index, 'Y') for index in x1) + \
                t_z, coefficient_2)
            transformed_term *= transf_op_1 + transf_op_2
 
        transf_op += transformed_term
    return transf_op