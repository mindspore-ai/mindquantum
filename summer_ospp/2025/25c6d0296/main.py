import os
import json
import time
import numpy as np
from scripts import *


def run_bifurcation(M_rows: int, N_cols: int):
    problem_params = {'M_rows': M_rows, 'N_cols': N_cols, 'A': 1.0}
    solver_params = {'ds': 0.002, 'max_active_paths': 10, 'max_total_steps': 100000, 'max_solver_steps': 100000}

    problem_name = f"grid_hubo_{problem_params['M_rows']}x{problem_params['N_cols']}"
    save_dir = f'results/{problem_name}'
    os.makedirs(save_dir, exist_ok=True)

    problem_instance = HuboProblem(**problem_params)
    t_start = time.time()

    np.random.seed(43)
    x_initial = np.random.randn(problem_instance.n_vars)
    manager = BifurcationManager(problem_instance, x_initial, **solver_params)
    records = manager.run(max_bifurcations=5)
    t_cons = time.time() - t_start
    if not records:
        pass
    else:
        bif_points_found = len(manager.bifurcation_points)

        with open(f"{save_dir}/route.json", "w") as f:
            json.dump(records, f, indent=4, cls=NumpyArrayEncoder)

        final_info = {"status": "Single run complete", **problem_params,
                      "t_seconds": t_cons,
                      "bifurcations_num": bif_points_found
                      }
        with open(f"{save_dir}/info.json", "w") as f:
            json.dump(final_info, f, indent=4)

def plot_bifurcation(M_rows: int, N_cols: int):
    # Define data file paths
    route_path = fr"results/grid_hubo_{M_rows}x{N_cols}/route.json"
    info_path = os.path.join(os.path.dirname(route_path), 'info.json')
    dir_path = os.path.dirname(route_path)
    # Load data from files
    tree_data = load_json(route_path)
    with open(info_path, 'r', encoding='utf-8') as f:
        info_data = json.load(f)

    # If data is loaded successfully, proceed with visualization.
    if tree_data:
        # Plot bifurcation diagram (p vs. a selected x variable).
        plot_bifurcation_diagram(tree_data, save_path=f"{dir_path}/bifurcation.png")
        # 1. Generate the plot colored by trajectory (cumulative arc length).
        visualize_pca_trajectory(tree_data, f'{dir_path}/pca_trajectory.png')
        # 2. Generate the plot colored by the evolution parameter 'p'.
        visualize_pca_p_colored(tree_data, f'{dir_path}/pca_p_colored.png')
        # Plot the energy landscape.
        problem = bifurcation.HuboProblem(M_rows=info_data['M_rows'], N_cols=info_data['N_cols'], A=info_data['A'])
        calculate_energy_history(tree_data, problem)
        plot_energy_landscape(tree_data, f'{dir_path}/energy_landscape.png')

def run_symmetry_comparison(grid_x_start, grid_x_end):
    def create_grid_list(start: int, stop: int):
        # (3, 5) -> [(3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]
        return [(i, i) for i in range(start, stop + 1) for _ in (0, 1)][: -1]

    grid_list = create_grid_list(grid_x_start, grid_x_end)
    for M_rows, N_cols in grid_list:
        print(f'--------------')
        print(f'Symmetry optimized method start on {M_rows}*{N_cols} grid...')
        t_start_wegner = time.time()
        solution_sym, energy_sym = run_lgt_problem_wegner(M_rows, N_cols)
        t_end_wegner = time.time() - t_start_wegner
        print(f'Symmetry optimized method end, time consumption: {t_end_wegner}')
        print(f'Classical method start on {M_rows}*{N_cols} grid...')
        t_start_classical = time.time()
        solution_classical, energy_classical = run_lgt_problem_classical(M_rows, N_cols)
        t_end_classical = time.time() - t_start_classical
        print(f'Classical method end, time consumption: {t_end_classical}')
        print(f'--------------')
        info = {
            'M_rows': M_rows,
            'N_cols': N_cols,
            'solution_sym': solution_sym,
            'energy_sym': energy_sym,
            'duration_sym': t_end_wegner,
            'solution_lqa': solution_classical,
            'energy_lqa': energy_classical,
            'duration_lqa': t_end_classical
        }
        os.makedirs('results', exist_ok=True)
        json_filepath = f"results/rows{M_rows}_cols{N_cols}.json"
        with open(json_filepath, 'w') as json_file:
            json.dump(info, json_file, indent=4)

    plot_symmetry(grid_x_start, grid_x_end, 'results')



if __name__ == '__main__':
    # # symmetry usage
    # M_rows, N_cols = 34, 35
    # t_start_wegner = time.time()
    # solution_sym, energy_sym = run_lgt_problem_wegner(M_rows, N_cols)
    # t_end_wegner = time.time() - t_start_wegner
    # info = {
    #     'M_rows': M_rows,
    #     'N_cols': N_cols,
    #     'solution_sym': solution_sym,
    #     'energy_sym': energy_sym,
    #     'duration_sym': t_end_wegner,
    # }
    # os.makedirs('results', exist_ok=True)
    # json_filepath = f"results/rows{M_rows}_cols{N_cols}.json"
    # with open(json_filepath, 'w') as json_file:
    #     json.dump(info, json_file, indent=4)

    # # symmetry plot
    # run_symmetry_comparison(3,33)

    # run bifurcation
    M_rows, N_cols = 32, 32
    # run_bifurcation(M_rows, N_cols)
    plot_bifurcation(M_rows, N_cols)




