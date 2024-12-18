#!/bin/bash


## Cora dataset
python3 grand.py --adjoint --dataset Cora --method dopri5 --adjoint_method dopri5 --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005

## CoauthorCS
python3 grand.py --adjoint --dataset CoauthorCS --method dopri5 --adjoint_method dopri5 --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005

## Photo
python3 grand.py --adjoint --dataset Photo --method dopri5 --adjoint_method dopri5 --block constant --function transformer --max_nfe 1000000 --step_size 0.005 --adjoint_step_size 0.005

