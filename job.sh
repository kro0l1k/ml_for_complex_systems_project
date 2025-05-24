#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --gres=gpumem:4096m
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=1024

module purge
module load stack/2024-05
module load gcc/13.2.0
module load python_cuda/3.11.6
source ~/venvs/ml_gpu_env/bin/activate
echo ">>> Now launching the Python script"

srun python -u jump_process_small_data_different_x0.py