#!/bin/sh
#COBALT -n 1
#COBALT -t 360
#COBALT -q gpu_a100
maxepochs=200
outputdir="runs64_01_19"
mkdir -p ${outputdir}
# methods=( "rk4" "dopri5" )
methods=( "rk4" )
for method in "${methods[@]}"
do
  python3 KS_node.py --node_model snode --max_epochs ${maxepochs} --train_dir ./${outputdir}/snode_${method} --double_prec --data_temporal_stride 10 --data_size 10000 --batch_size 256 --node_method ${method} --step_size 0.001 --time_window_size 1 --tb_log
done
