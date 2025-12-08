import sys
import os

sys.path.append('mlatom')
import mlatom as ml

nstates = 3
initial_state = 2
temperatures = [0]
number_of_initial_conditions=50


def main():
    # 初始化 MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    if rank == 0:


        mol = ml.data.molecule()
        mol.read_from_xyz_file('./materials/h3_angstrom.xyz')
        mol.charge = 1
        mol.multiplicity = 1
        print('done read original xyz')

        # 定义计算方法
        method_optfreq = ml.models.methods(method='CASCI-SINGLET/sto-3g', program='pyscf')
        # pyscf_geom = ml.models.methods(
        #     method='CASCI-SINGLET/sto-3g',
        #     program='pyscf',
        #     qm_program_kwargs={'save_files_in_current_directory': True}
        # )

        # # 优化几何
        # geomopt = ml.simulations.optimize_geometry(
        #     model=pyscf_geom,
        #     program='pyscf',
        #     initial_molecule=mol
        # )
        eqmol = mol
        # eqmol.write_file_with_xyz_coordinates('opt_geom_c2h4_casci_singlet.xyz')
        # print('done opt geom and save')
        # ml.simulations.freq(molecule=eqmol, model=method_optfreq, program='pyscf')
        # eqmol.dump(filename='freq_h3p_casci_singlet.json', format='json')
        # print('end freq opt')

    else:
        eqmol = None
    eqmol = comm.bcast(eqmol, root=0)
    comm.Barrier()

    for temp in temperatures:
        directory_name = f'namd_temp_h3p_{temp}K_initial_state_{initial_state}_{number_of_initial_conditions}_vqe_qseEX_hft_singlet_02time_curvatureblock'


        if rank == 0:
            os.system(f'rm -rf {directory_name}; mkdir {directory_name}')
        comm.Barrier()


        os.chdir(directory_name)


        if rank == 0:
            # init_cond_db = ml.generate_initial_conditions(
            #     molecule=eqmol,
            #     generation_method='wigner',
            #     initial_temperature=temp,
            #     number_of_initial_conditions=number_of_initial_conditions,random_seed=123
            # )
            # init_cond_db.dump(f'init_cond_h3p_{temp}K.json', format='json')


            eqmol=ml.data.load_return_molecule('../materials/freq_h3p_casci_singlet.json',format='json')
            init_cond_db=ml.generate_initial_conditions(file_with_initial_xyz_velocities='../materials/init_cond_h3p_0K.json',file_with_initial_xyz_coordinates='../materials/init_cond_h3p_0K.json',initial_temperature=temp,number_of_initial_conditions=number_of_initial_conditions,generation_method='from-json',molecule=eqmol)

        else:
            init_cond_db = None
        init_cond_db = comm.bcast(init_cond_db, root=0)
        comm.Barrier()

    
        pyscf1 = ml.models.methods(
            method='CASCI-VQE-EX/sto-3g',
            program='pyscf',
            qm_program_kwargs={'save_files_in_current_directory': True}
        )

     
        maxium_propa_time = 10
        time_step = 0.2
        namd_kwargs = {
            'model': pyscf1,
            'time_step': time_step,
            'maximum_propagation_time': maxium_propa_time,
            'hopping_algorithm': 'LZBL',
            'nstates': nstates,
            'initial_state': initial_state
        }

       
        dyns = ml.simulations.run_in_parallel_mpi(
            molecular_database=init_cond_db,
            task=ml.namd_block_adj.surface_hopping_md,
            task_kwargs=namd_kwargs,
            create_temp_directories=True
        )

        comm.Barrier()

        # trajs = []
        # itraj = 0
        # for d in dyns:
        #     if d is not None and hasattr(d, 'molecular_trajectory'):
        #         traj = d.molecular_trajectory
        #         trajs.append(traj)
        #         itraj += 1
        #         traj.dump(filename=f"{maxium_propa_time}X{time_step}Xhtraj{itraj}.h5", format='h5md')
        #     else:
        #         print(f"Skipping failed dyn at index {itraj}")
        

 
        if rank == 0:
            # trajs = [d.molecular_trajectory for d in dyns]
            # itraj = 0
            # for traj in trajs:
            #     itraj += 1
            #     traj.dump(filename=f"{maxium_propa_time}X{time_step}Xhtraj{itraj}.h5", format='h5md')

            trajs = []
            itraj = 0
            for d in dyns:
                if d is not None and hasattr(d, 'molecular_trajectory'):
                    traj = d.molecular_trajectory
                    trajs.append(traj)
                    itraj += 1
                    traj.dump(filename=f"{maxium_propa_time}X{time_step}Xhtraj{itraj}.h5", format='h5md')
                else:
                    print(f"Skipping failed dyn at index {itraj}")

            ml.namd_block_adj.analyze_trajs(
                trajectories=trajs,
                maximum_propagation_time=maxium_propa_time
            )
            ml.namd_block_adj.plot_population(
                trajectories=trajs,
                time_step=time_step,
                max_propagation_time=maxium_propa_time,
                nstates=nstates,
                filename='pop.png',
                pop_filename='pop.txt'
            )

      
        os.chdir('..')
        comm.Barrier()


if __name__ == '__main__':
    main()