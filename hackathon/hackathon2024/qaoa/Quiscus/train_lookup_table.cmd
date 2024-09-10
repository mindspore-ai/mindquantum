:ft_ada_decay
python train_lookup_table.py --name ft-ada-decay

:ft_ada_decay_moment_fast
python train_lookup_table.py --name ft-ada-decay-moment-fast --momentum 0.6 --steps 40

:ft_ada_decay_moment_fast_ft
REM resume from fad-9400, finetune 1k step
python train_lookup_table.py --name ft-ada-decay-moment-fast_ft --momentum 0.6 --steps 40 --load log\ft-ada-decay\lookup_table-iter=9400.json --iters 10400

:ft_ada_decay_moment_fast_ex
REM resume from fad-9400, finetune 3k step
python train_lookup_table.py --name ft-ada-moment-fast_ft-ex --dx_decay 1.0 --momentum 0.6 --steps 40 --load_base log\ft-ada-decay\lookup_table-iter=9400.json --iters 12400 --rescaler 1.165 --ex
