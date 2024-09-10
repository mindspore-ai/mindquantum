@ECHO OFF

REM classical baselines

python run_baseline.py -M linear -E zf -D maxlog
python run_baseline.py -M linear -E zf -D app
python run_baseline.py -M linear -E mf -D maxlog
python run_baseline.py -M linear -E mf -D app
python run_baseline.py -M linear -E lmmse -D maxlog
python run_baseline.py -M linear -E lmmse -D app

python run_baseline.py -M kbest -k 16
python run_baseline.py -M kbest -k 32
python run_baseline.py -M kbest -k 64

python run_baseline.py -M ep -l 5
python run_baseline.py -M ep -l 10
python run_baseline.py -M ep -l 20

python run_baseline.py -M mmse --num_iter 1
python run_baseline.py -M mmse --num_iter 2
python run_baseline.py -M mmse --num_iter 4
python run_baseline.py -M mmse --num_iter 8
python run_baseline.py -M mmse --num_iter 16
