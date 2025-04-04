#!/bin/bash

python trainer.py --epochs 100 --ndata 1000 --batch 5 --lr 1e-4 --eq_res 1e-5 --gamma 0.98 --last_lr 1e-5 --loss_m l2 --method ConFIG --seed 1234
#python trainer.py --epochs 1000 --ndata 3000 --batch 5 --lr 1e-4 --eq_res 1e-5 --gamma 0.98 --last_lr 1e-5 --device 2 --loss_m l2 --method ConFIG

# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 2e-4 --gamma 0.98 --last_lr 1e-5 --device 2 --eq_res 1e-5 --method ConFIG --loss_m l2
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 2e-4 --gamma 0.98 --last_lr 1e-5 --device 2 --eq_res -1   --method ConFIG --loss_m l2
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 2 --eq_res 1e-4 --method PINN   --loss_m l2
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 5e-5 --gamma 0.98 --last_lr 1e-5 --device 2 --eq_res 5e-5 --method PINN   --loss_m l2

# python trainer.py --epochs 1000 --ndata 3000 --batch 5 --lr 2e-4 --gamma 0.98 --last_lr 1e-5 --device 2 --loss_m l1 --method ConFIG

# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 2e-4 --gamma 0.98 --last_lr 1e-5 --device 2 --loss_m l1 --method ConFIG
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 8e-5 --gamma 0.98 --last_lr 1e-5 --device 2 --loss_m l1 --method ConFIG
