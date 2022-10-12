# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
from src.dataset import *
from src.maxcut import maxcut
def _test(n, depth, problem, grad, show_iter_val):
    if grad:
        opti_args = dict(method='bfgs',
                         jac=True,
                         options={'gtol': 1e-3}
                         )
    else:
        opti_args = dict(method='Nelder-Mead',
                         tol=1e-3
                         )
    _, _, res = maxcut(n, depth, problem,
                       grad=grad, show_iter_val=True,
                       **opti_args)
    return res
def test(dataset, build_dataset, depth, grad):
    """Test."""
    n, problem = build_dataset()
    print(f'dataset: {dataset}')
    print(f'n: {n}')
    print(f'n_qubits: {(n + 1) // 2}')
    print(f'depth: {depth}')
    print(f'grad: {grad}')
    res = _test(n, depth, problem, grad, True)
    print('result:', res[:n])
    print('score:', score(problem, res))
def test_parallel(dataseta, build_dataseta,
                  datasetb, build_datasetb,
                  depth, grad):
    """Test."""
    p1 = build_dataseta()
    p2 = build_datasetb()
    n, problem, od = build_dataset_parallel(*p1, *p2)
    m = (n + 1) // 2
    ds = [dataseta, datasetb]
    pb = [p1, p2]
    print(f'dataset: {ds[od[0]]} & {ds[od[1]]}')
    print(f'n: {n}')
    print(f'n_qubits: {m}')
    print(f'depth: {depth}')
    print(f'grad: {grad}')
    res = _test(n, depth, problem, grad, True)
    print(f'dataset: {ds[od[0]]}')
    print('result:', res[:pb[od[0]][0]])
    print('score:', score(pb[od[0]][1], res[:m]))
    print(f'dataset: {ds[od[1]]}')
    print('result:', res[m:m+pb[od[1]][0]])
    print('score:', score(pb[od[1]][1], res[m:]))
if __name__ == '__main__':
    parallel = False
    if not parallel:
        test('dataset1', build_dataset1, 7, False)
        print('\n----- - ----- - ----- - -----\n')
        test('dataset1', build_dataset1, 7, True)
        print('\n----- - ----- - ----- - -----\n')
        test('dataset2', build_dataset2, 4, True)
        print('\n----- - ----- - ----- - -----\n')
        test('dataset3', build_dataset3, 4, True)
    else:
        test_parallel('dataset1', build_dataset1,
                      'dataset2', build_dataset2,
                      4, True)
        print('\n----- - ----- - ----- - -----\n')
        test_parallel('dataset2', build_dataset2,
                      'dataset3', build_dataset3,
                      4, True)
