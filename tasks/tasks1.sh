#!/bin/bash

python trainer.py --epochs 1000 --ndata 3000 --batch 5 --lr 1e-4 --eq_res -1 --gamma 0.98 --last_lr 1e-5 --device 1 --loss_m l2 --method ConFIG --length 1 --checkpoint ../178/checkpoint225.pt

# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 1 --eq_res -1 --method ConFIG --loss_m l2 --length 1
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 1 --eq_res -1 --method ConFIG --loss_m l2 --length 0

# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-4 --device 1 --eq_res 1e-5 --method PINN   --loss_m l2
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 5e-5 --gamma 0.98 --last_lr 5e-5 --device 1 --eq_res -1   --method PINN   --loss_m l2
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 2e-4 --gamma 0.98 --last_lr 1e-5 --device 1 --eq_res 1e-5 --method ConFIG --loss_m l2
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 2e-4 --gamma 0.98 --last_lr 1e-5 --device 1 --eq_res -1   --method ConFIG --loss_m l2


# python trainer.py --epochs 1000 --ndata 3000 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 1 --loss_m l1 --method PINN --eq_res 1e-5
# python trainer.py --epochs 1000 --ndata 3000 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 2 --loss_m l1 --method std

# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 5e-6 --device 1 --loss_m l1 --method ConFIG
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-6 --device 1 --loss_m l1 --method ConFIG
# python trainer.py --epochs 1000 --ndata 3000 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 2 --loss_m l1 --method ConFIG

# Params search
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 5e-5 --gamma 0.98 --last_lr 1e-5 --device 2 --method PINN --eq_res 1e-5
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 5e-5 --gamma 0.98 --last_lr 1e-5 --device 2 --method PINN --eq_res 5e-5
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 2 --method PINN --eq_res 1e-5
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 2 --method PINN --eq_res 1e-4

# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 5e-4 --gamma 0.98 --last_lr 1e-5 --device 2 --method ConFIG
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 2 --method ConFIG
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 5e-5 --gamma 0.98 --last_lr 1e-5 --device 2 --method ConFIG
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-5 --gamma 0.98 --last_lr 1e-5 --device 2 --method ConFIG
