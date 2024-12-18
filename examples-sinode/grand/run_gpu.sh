#!/bin/bash


## Cora dataset
python3 grand.py --adjoint --dataset Cora --method euler --adjoint_method euler --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005
python3 grand.py --adjoint --dataset Cora --method midpoint --adjoint_method midpoint  --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005
python3 grand.py --adjoint --dataset Cora --method rk4 --adjoint_method rk4 --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005
python3 grand.py --adjoint --dataset Cora --method explicit_adams --adjoint_method explicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005
python3 grand.py --adjoint --dataset Cora --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 0.1 --filename_suffix _1e-1
python3 grand.py --adjoint --dataset Cora --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 0.25 --filename_suffix _25e-2
python3 grand.py --adjoint --dataset Cora --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 0.5 --filename_suffix _5e-1
python3 grand.py --adjoint --dataset Cora --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 1 --filename_suffix _1
python3 grand.py --adjoint --dataset Cora --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 2 --filename_suffix _2
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _l2_1e-1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _l2_25e-2 
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _l2_5e-1 
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 1 --filename_suffix _l2_1 
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 2 --filename_suffix _l2_2
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _3_1e-1 
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _3_25e-2 
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _3_5e-1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 1 --filename_suffix _3_1 
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 2 --filename_suffix _3_2 
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _4_1e-1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _4_25e-2
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _4_5e-1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 1 --filename_suffix _4_1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 2 --filename_suffix _4_2
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _5_1e-1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _5_25e-2
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _5_5e-1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 1 --filename_suffix _5_1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 2 --filename_suffix _5_2
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _1e-1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _25e-2
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _5e-1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 1 --filename_suffix _1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 2 --filename_suffix _2
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _1e-1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _25e-2
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _5e-1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 1 --filename_suffix _1
python3 grand.py --adjoint --dataset Cora --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 2 --filename_suffix _2


## CoauthorCS
python3 grand.py --adjoint --dataset CoauthorCS --method euler --adjoint_method euler --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005
python3 grand.py --adjoint --dataset CoauthorCS --method midpoint --adjoint_method midpoint  --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005
python3 grand.py --adjoint --dataset CoauthorCS --method rk4 --adjoint_method rk4 --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005
python3 grand.py --adjoint --dataset CoauthorCS --method explicit_adams --adjoint_method explicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005
python3 grand.py --adjoint --dataset CoauthorCS --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 0.1 --filename_suffix _1e-1
python3 grand.py --adjoint --dataset CoauthorCS --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 0.25 --filename_suffix _25e-2
python3 grand.py --adjoint --dataset CoauthorCS --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 0.5 --filename_suffix _5e-1
python3 grand.py --adjoint --dataset CoauthorCS --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 1 --filename_suffix _1
python3 grand.py --adjoint --dataset CoauthorCS --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 2 --filename_suffix _2
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _l2_1e-1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _l2_25e-2 
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _l2_5e-1 
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 1 --filename_suffix _l2_1 
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 2 --filename_suffix _l2_2
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _3_1e-1 
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _3_25e-2 
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _3_5e-1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 1 --filename_suffix _3_1 
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 2 --filename_suffix _3_2 
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _4_1e-1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _4_25e-2
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _4_5e-1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 1 --filename_suffix _4_1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 2 --filename_suffix _4_2
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _5_1e-1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _5_25e-2
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _5_5e-1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 1 --filename_suffix _5_1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 2 --filename_suffix _5_2
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _1e-1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _25e-2
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _5e-1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 1 --filename_suffix _1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 2 --filename_suffix _2
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _1e-1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _25e-2
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _5e-1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 1 --filename_suffix _1
python3 grand.py --adjoint --dataset CoauthorCS --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 2 --filename_suffix _2

## Photo
python3 grand.py --adjoint --dataset Photo --method euler --adjoint_method euler --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005
python3 grand.py --adjoint --dataset Photo --method midpoint --adjoint_method midpoint  --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005
python3 grand.py --adjoint --dataset Photo --method rk4 --adjoint_method rk4 --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005
python3 grand.py --adjoint --dataset Photo --method explicit_adams --adjoint_method explicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005
python3 grand.py --adjoint --dataset Photo --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 0.1 --filename_suffix _1e-1
python3 grand.py --adjoint --dataset Photo --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 0.25 --filename_suffix _25e-2
python3 grand.py --adjoint --dataset Photo --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 0.5 --filename_suffix _5e-1
python3 grand.py --adjoint --dataset Photo --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 1 --filename_suffix _1
python3 grand.py --adjoint --dataset Photo --method implicit_adams --adjoint_method implicit_adams --block constant --function transformer --max_nfe 1000000 --step_size 2 --filename_suffix _2
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _l2_1e-1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _l2_25e-2 
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _l2_5e-1 
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 1 --filename_suffix _l2_1 
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type l2 -snes_type ksponly -ksp_rtol 1e-4 --step_size 2 --filename_suffix _l2_2
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _3_1e-1 
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _3_25e-2 
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _3_5e-1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 1 --filename_suffix _3_1 
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 3 -snes_type ksponly -ksp_rtol 1e-4 --step_size 2 --filename_suffix _3_2 
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _4_1e-1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _4_25e-2
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _4_5e-1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 1 --filename_suffix _4_1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 4 -snes_type ksponly -ksp_rtol 1e-4 --step_size 2 --filename_suffix _4_2
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _5_1e-1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _5_25e-2
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _5_5e-1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 1 --filename_suffix _5_1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --imex --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ts_arkimex_type 5 -snes_type ksponly -ksp_rtol 1e-4 --step_size 2 --filename_suffix _5_2
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _1e-1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _25e-2
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _5e-1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 1 --filename_suffix _1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --max_nfe 1000000 --method cn --adjoint_method cn --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 2 --filename_suffix _2
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.1 --filename_suffix _1e-1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.25 --filename_suffix _25e-2
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 0.5 --filename_suffix _5e-1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 1 --filename_suffix _1
python3 grand.py --adjoint --dataset Photo --block pnode --function mytransformer --max_nfe 1000000 --method beuler --adjoint_method beuler --implicit_form --use_dlpack -ts_trajectory_type memory -ts_adapt_type none -ksp_rtol 1e-4 --step_size 2 --filename_suffix _2
