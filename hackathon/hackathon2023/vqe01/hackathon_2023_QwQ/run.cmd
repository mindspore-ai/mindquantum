REM run specified optimizer and initializer
python solver.py -O BFGS         -I eq-2d
python solver.py -O trust-constr -I randu


REM grid search over optimizer & initializer
python solver.py --run_all --log_path log.pyscf --objective pyscf
python solver.py --run_all --log_path log.uccsd --objective uccsd

python vis_fci_cmp.py --log_path log.pyscf
python vis_fci_cmp.py --log_path log.uccsd


REM run solution for h4.csv (cheaty)
python solver.py --objective pyscf -O trust-constr --init randn --no_comp
python solver.py --check

REM run solution for h4.csv (formal)
python solver.py
python solver.py --check
