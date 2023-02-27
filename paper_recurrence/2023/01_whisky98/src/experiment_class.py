from src.iteration_generators import Generate_iterations
from src.koopman_iteration_generators import Generate_koopman_iterations
import matplotlib.pyplot as plt
import pickle

class Koopman_iteration_accelerator():
    # designed for a complete experiment with one it_generator and several koopman_generators

    def __init__(self,experiment_name):
        self.koopman_generators = {}
        self.x_results = {}
        self.value_results = {}
        self.koopman_infos = {}
        self.experiment_name = experiment_name

    def set_it_generator(self,one_step_iteration_generator,several_steps_iteration_generator):
        # one of the two args is None and the other is callable
        self.it_generator,self.its_generator = one_step_iteration_generator,several_steps_iteration_generator
        print('set_it_generator is executed.')

    def set_koopman_generator(self,name,one_step_koopman_iteration_generator):
        self.koopman_generators[name] = one_step_koopman_iteration_generator
        print('set_koopman_generator is executed.')

    def set_shared_args(self,x_init,num_iters,num_pieces,value_fn,args_1):
        self.x_init = x_init
        self.num_iters = num_iters
        self.num_pieces = num_pieces
        self.value_fn = value_fn
        self.args_1 = args_1
        print('set_shared_args is executed.')

    def start_accelerated_iteration(self,name,args_2):
        # input: arguments for it_generator and koopman_generator, name string for the result vector
        # output: np array of the result of the whole iteration and save it to self.results
        x_init,num_iters,num_pieces = self.x_init,self.num_iters,self.num_pieces
        one_step_iteration_generator,several_steps_iteration_generator,one_step_koopman_iteration_generator = self.it_generator,self.its_generator,self.koopman_generators[name]
        args_1 = self.args_1
        value_fn = self.value_fn
        x_results = [x_init]
        value_results = [self.value_fn(x_init)]
        koopman_infos = []

        for i in range(num_pieces):
            iterations_generated,value_iterations_generated = Generate_iterations(one_step_iteration_generator,several_steps_iteration_generator,x_init,num_iters,*args_1)
            koopman_pred,value_koopman_pred,koopman_info = Generate_koopman_iterations(one_step_koopman_iteration_generator,iterations_generated,value_fn,*args_2)
            x_results = x_results+iterations_generated
            x_results.append(koopman_pred)
            value_results = value_results+value_iterations_generated
            value_results.append(value_koopman_pred)
            x_init = koopman_pred
            koopman_infos.append(koopman_info)
            print(f'start_accelerated_iteration-----{i+1}/{num_pieces}')
        print('start_accelerated_iteration is executed.')

        self.x_results[name] = x_results
        self.value_results[name] = value_results
        self.koopman_infos[name] = koopman_infos

    def start_raw_iteration(self):
        num_all_iterations = self.num_pieces*(self.num_iters+1)+1
        one_step_iteration_generator,several_steps_iteration_generator = self.it_generator,self.its_generator
        x_now = self.x_init
        args_1 = self.args_1
        x_results = [x_now]
        value_results = [self.value_fn(x_now)]
        result_name = 'raw'

        if callable(one_step_iteration_generator):
            for _ in range(num_all_iterations-1):
                x_next,value_now = one_step_iteration_generator(x_now,*args_1)
                x_results.append(x_next)
                value_results.append(value_now)
                x_now = x_next
        else:
            x_results_temp,value_results_temp = several_steps_iteration_generator(x_now,num_all_iterations-1,*args_1)
            x_results = x_results+x_results_temp
            value_results = value_results+value_results_temp

        self.x_results[result_name] = x_results
        self.value_results[result_name] = value_results
        print('start_raw_iteration is executed.')

    def plot_value_results(self):
        num_time_steps = len(self.x_results['raw'])
        x = range(num_time_steps)

        for name,value in self.value_results.items():
            plt.plot(x,value,label=name)
        plt.title(self.experiment_name)

        plt.legend()
        plt.show()
        print('plot_value_results is executed.')

    def save_experiment(self,save_message):
        file_name = '..\\results\\'+self.experiment_name+save_message+'.txt'
        f = open(file_name,'wb')
        pickle.dump(self,f)
        f.close()
        print('save_experiment is executed.')

    def load_experiment(self,path):
        f = open(path,'rb')
        self.__dict__ = pickle.load(f).__dict__
        print(self.args_1)
        f.close()
        print('load_experiment is executed.')


