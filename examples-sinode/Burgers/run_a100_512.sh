#!/bin/bash
#COBALT -n 1
#COBALT -t 360
#COBALT -q gpu_a100

# set up the path for petsc4py
export PYTHONPATH=$PETSC_DIR/$PETSC_ARCH/lib

### Burger Eqn
outputdir="runs512"
bsize=200
nepoch=20
myss=1e-3
python3 Burgers.py --pnode -ts_trajectory_type memory --use_dlpack --double_prec -ts_adapt_type none --method fixed_dopri5 --step_size $myss --epoch $nepoch --batch_size $bsize --batch_time 1 --test_freq 1 --tb_log --train_dir ./${outputdir}_snode_dopri5
python3 Burgers.py --pnode -ts_trajectory_type memory --use_dlpack --double_prec --method rk4 --step_size $myss --epoch $nepoch --batch_size $bsize --batch_time 1 --test_freq 1 --tb_log --train_dir ./${outputdir}_snode_rk4
python3 Burgers.py --double_prec --method explicit_adams --step_size $myss --epoch $nepoch --batch_size $bsize --batch_time 1 --test_freq 1 --tb_log --train_dir ./${outputdir}_snode_explicit_adams
python3 Burgers.py --double_prec --method implicit_adams --epoch $nepoch --batch_size $bsize --batch_time 1 --test_freq 1 --tb_log --train_dir ./${outputdir}_snode_implicit_adams

##torch
python3 Burgers.py --adjoint --pnode --double_prec --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly --linear_solver torch --epoch $nepoch --batch_size $bsize --batch_time 1 --test_freq 1 --tb_log --train_dir ./${outputdir}_imex_l2
python3 Burgers.py --adjoint --pnode --double_prec --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly --linear_solver torch --epoch $nepoch --batch_size $bsize --batch_time 1 --test_freq 1 --tb_log --train_dir ./${outputdir}_imex_3
python3 Burgers.py --adjoint --pnode --double_prec --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly --linear_solver torch --epoch $nepoch --batch_size $bsize --batch_time 1 --test_freq 1 --tb_log --train_dir ./${outputdir}_imex_4
python3 Burgers.py --adjoint --pnode --double_prec --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly --linear_solver torch --epoch $nepoch --batch_size $bsize --batch_time 1 --test_freq 1 --tb_log --train_dir ./${outputdir}_imex_5

##torch
python3 Burgers.py --adjoint --pnode --double_prec --method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none --linear_solver petsc --epoch $nepoch --batch_size $bsize --batch_time 1 --test_freq 1 --tb_log --train_dir ./${outputdir}_snode_cn
python3 Burgers.py --adjoint --pnode --double_prec --method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none --linear_solver petsc --epoch $nepoch --batch_size $bsize --batch_time 1 --test_freq 1 --tb_log --train_dir ./${outputdir}_snode_beuler

