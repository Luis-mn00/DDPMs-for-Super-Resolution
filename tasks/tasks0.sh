#!/bin/bash

#python trainer.py --epochs 1000 --ndata 3000 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --loss_m l2 --method ConFIG --length 1
python trainer.py --dataset 2 --epochs 1000 --ndata 5000 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --loss_m l2 --method ConFIG --length 1

#python trainer.py --epochs 1000 --ndata 3000 --batch 5 --lr 1e-4 --eq_res 1e-5 --gamma 0.98 --last_lr 1e-5 --device 0 --loss_m l2 --method PINN

# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 0 --eq_res 1e-5 --method PINN   --loss_m l2
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 0 --eq_res -1   --method PINN   --loss_m l2
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 0 --eq_res 1e-5 --method ConFIG --loss_m l2
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 0 --eq_res -1   --method ConFIG --loss_m l2

# python trainer.py --epochs 400 --ndata 200 --batch 5 --lr 0.0001 --gamma 0.98 --last_lr 1e-5 --device 0 --method multiConFIG --nmulti 2
# python trainer.py --epochs 400 --ndata 200 --batch 5 --lr 0.0001 --gamma 0.98 --last_lr 1e-5 --device 0 --method multiConFIG --nmulti 3
# python trainer.py --epochs 400 --ndata 200 --batch 5 --lr 0.0001 --gamma 0.98 --last_lr 1e-5 --device 0 --method multiConFIG --nmulti 4
# python trainer.py --epochs 400 --ndata 200 --batch 5 --lr 0.0001 --gamma 0.98 --last_lr 1e-5 --device 0 --method multiConFIG --nmulti 6
# python trainer.py --epochs 400 --ndata 200 --batch 5 --lr 0.0001 --gamma 0.98 --last_lr 1e-5 --device 0 --method multiConFIG --nmulti 8
# python trainer.py --epochs 400 --ndata 200 --batch 5 --lr 0.0001 --gamma 0.98 --last_lr 1e-5 --device 0 --method multiConFIG --nmulti 10
# python trainer.py --epochs 400 --ndata 200 --batch 5 --lr 0.0001 --gamma 0.98 --last_lr 1e-5 --device 0 --method multiConFIG --nmulti 16
# python trainer.py --epochs 400 --ndata 200 --batch 5 --lr 0.0001 --gamma 0.98 --last_lr 1e-5 --device 0 --method std

# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 5e-5 --gamma 0.98 --last_lr 1e-5 --device 0 --loss_m l1 --method PINN --eq_res 1e-5
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 5e-5 --gamma 0.98 --last_lr 1e-5 --device 0 --loss_m l1 --method PINN --eq_res 5e-5
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 0 --loss_m l1 --method PINN --eq_res 1e-5
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 0 --loss_m l1 --method PINN --eq_res 1e-4

# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 5e-4 --gamma 0.98 --last_lr 1e-5 --device 0 --loss_m l1 --method ConFIG
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-4 --gamma 0.98 --last_lr 1e-5 --device 0 --loss_m l1 --method ConFIG
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 5e-5 --gamma 0.98 --last_lr 1e-5 --device 0 --loss_m l1 --method ConFIG
# python trainer.py --epochs 200 --ndata 500 --batch 5 --lr 1e-5 --gamma 0.98 --last_lr 1e-5 --device 0 --loss_m l1 --method ConFIG
