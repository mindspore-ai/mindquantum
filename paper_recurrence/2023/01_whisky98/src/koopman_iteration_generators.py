from pydmd import DMD,HankelDMD
import numpy as np

def Generate_koopman_iterations(one_step_koopman_iteration_generator,iterations_generated,value_fn,*args):
    return one_step_koopman_iteration_generator(iterations_generated,value_fn,*args)


def dmd_iteration_generator_predict_from_future_steps(its_generated,value_fn,num_iters_koopman):
    num_iters = len(its_generated)
    dmd = DMD()
    data_matrix = np.array(its_generated).T

    dmd.fit(data_matrix)

    dmd.dmd_time['dt'] = 1
    dmd.dmd_time['t0'] = num_iters
    dmd.dmd_time['tend'] = num_iters+num_iters_koopman-1
    its_koopman = dmd.reconstructed_data.real
    values_koopman = [value_fn(its_koopman[:,i]) for i in range(num_iters_koopman)]

    optimal_index = values_koopman.index(min(values_koopman))
    koopman_info = {'dmd_model':dmd,'optimal_index':optimal_index}


    return its_koopman[:,optimal_index],values_koopman[optimal_index],koopman_info

def hankeldmd_iteration_generator_predict_from_future_steps(its_generated,value_fn,num_iters_koopman,order):
    num_iters = len(its_generated)
    hankeldmd = HankelDMD(d=order)
    data_matrix = np.array(its_generated).T
    hankeldmd.fit(data_matrix)

    hankeldmd.dmd_time['dt'] = 1
    hankeldmd.dmd_time['t0'] = num_iters
    hankeldmd.dmd_time['tend'] = num_iters+num_iters_koopman-1
    its_koopman = hankeldmd.reconstructed_data.real
    values_koopman = [value_fn(its_koopman[:,i]) for i in range(num_iters_koopman)]

    optimal_index = values_koopman.index(min(values_koopman))
    koopman_info = {'hankeldmd_model':hankeldmd,'optimal_index':optimal_index}

    return its_koopman[:,optimal_index].real,values_koopman[optimal_index],koopman_info