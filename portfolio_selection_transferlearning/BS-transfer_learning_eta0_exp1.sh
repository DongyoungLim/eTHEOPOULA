#!/bin/bash

#SBATCH --job-name=BS-transfer_learning_eta0_exp1
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


# num_asset = 5
python -u  main_transfer_learning.py --asset_model BS --num_asset 5 --num_path 20000 --u_gamma 4 --act_fn relu --hidden_size 1 --optimizer theopoula --lr 0.1 --eps 0.01 --beta 1e12 --batch_size 128 --epochs 100 --scheduler --scheduler_type step --eta 0
python -u  main_transfer_learning.py --asset_model BS --num_asset 5 --num_path 20000 --u_gamma 4 --act_fn relu --hidden_size 5 --optimizer theopoula --lr 0.1 --eps 0.01 --beta 1e12 --batch_size 128 --epochs 100 --scheduler --scheduler_type step  --eta 0
python -u  main_transfer_learning.py --asset_model BS --num_asset 5 --num_path 20000 --u_gamma 4 --act_fn relu --hidden_size 10 --optimizer theopoula --lr 0.1 --eps 0.01 --beta 1e12 --batch_size 128 --epochs 100 --scheduler --scheduler_type step --eta 0


# num_asset = 50
python -u  main_transfer_learning.py --asset_model BS --num_asset 50 --num_path 20000 --u_gamma 5 --act_fn relu --hidden_size 1 --optimizer theopoula --lr 0.05 --eps 0.0001 --beta 1e12 --batch_size 128 --epochs 100 --scheduler --scheduler_type step --eta 0
python -u  main_transfer_learning.py --asset_model BS --num_asset 50 --num_path 20000 --u_gamma 5 --act_fn relu --hidden_size 5 --optimizer theopoula --lr 0.05 --eps 0.0001 --beta 1e12 --batch_size 128 --epochs 100 --scheduler --scheduler_type step --eta 0
python -u  main_transfer_learning.py --asset_model BS --num_asset 50 --num_path 20000 --u_gamma 5 --act_fn relu --hidden_size 20 --optimizer theopoula --lr 0.05 --eps 0.0001 --beta 1e12 --batch_size 128 --epochs 100 --scheduler --scheduler_type step --eta 0



