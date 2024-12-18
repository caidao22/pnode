#!/bin/sh
#COBALT -n 1
#COBALT -t 360
#COBALT -q gpu_a100
# set up the path for petsc4py
export PYTHONPATH=$PETSC_DIR/$PETSC_ARCH/lib
maxepochs=200
outputdir="runs64"
mkdir -p ${outputdir}
methods=( "rk4" "fixed_dopri5" )
for method in "${methods[@]}"
do
  python3 KS.py --pnode_model snode -ts_trajectory_type memory --max_epochs ${maxepochs} --train_dir ./${outputdir}/snode_${method} --double_prec --data_temporal_stride 10 --data_size 50000 --batch_size 256 --pnode_method ${method} --step_size 0.001 --time_window_size 1 --tb_log
done
methods=( "cn" ) 
for method in "${methods[@]}"
do
  python3 KS.py --pnode_model snode -ts_trajectory_type memory --max_epochs ${maxepochs} --train_dir ./${outputdir}/snode_${method} --double_prec --data_temporal_stride 10 --data_size 50000 --batch_size 256 --pnode_method ${method} --step_size 0.2 --time_window_size 1 --tb_log -ts_adapt_type none --implicit_form
done
methods=( "l2" "3" "4" "5" )
for method in "${methods[@]}"
do
  python3 KS.py --pnode_model imex -ts_trajectory_type memory --max_epochs ${maxepochs} --train_dir ./${outputdir}/imex_${method} --double_prec --data_temporal_stride 10 --data_size 50000 --batch_size 256 --step_size 0.2 --time_window_size 1 --tb_log --linear_solver torch -ts_adapt_type none -snes_type ksponly -ts_arkimex_type ${method}
done
