#!/bin/bash

#SBATCH --job-name=AR-BEST-THEOPOULA
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=sc075
#SBATCH --output=./outputs/output-%x.out
#SBATCH --error=./errs/error-%x.err

module load gcc/8.2.0
module load nvidia/nvhpc
module load nvidia/nvhpc-nompi/22.2
module load nvidia/cudnn/8.2.1-cuda-11.6
module load openmpi/4.1.2-cuda-11.6
module load mpi4py/3.1.3-ompi-gpu


module load horovod/0.24.2-gpu

python -u main.py --asset_model AR --num_asset 30 --num_path 40000 --u_gamma 15 --num_step 10 --r_f 1.03 --hidden_size 5 --optimizer theopoula --lr 0.01 --eps 1e-12 --beta 1e12 --batch_size 128 --epochs 100 --scheduler --scheduler_type step
python -u main.py --asset_model AR --num_asset 30 --num_path 40000 --u_gamma 15 --num_step 10 --r_f 1.03 --hidden_size 20 --optimizer theopoula --lr 0.01 --eps 1e-8 --beta 1e12 --batch_size 128 --epochs 100 --scheduler --scheduler_type step
python -u main.py --asset_model AR --num_asset 30 --num_path 40000 --u_gamma 15 --num_step 10 --r_f 1.03 --hidden_size 50 --optimizer theopoula --lr 0.01 --eps 0.01 --beta 1e12 --batch_size 128 --epochs 100 --scheduler --scheduler_type step

