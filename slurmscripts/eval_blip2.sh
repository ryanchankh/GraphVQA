#!/bin/bash
#SBATCH --partition=gpu           # Specify the partition or queue name
#SBATCH --nodes=1
#SBATCH --mem=150G                # Memory to allocate (in GB) #SBATCH
#SBATCH --output=/shared_data/p_vidalr/ryanckh/GraphVQA/slurmlog/%j.out
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=96:00:00                 # Maximum runtime (format: HH:MM:SS)

# Load any necessary modules
module load cuda/12.2          # Load Anaconda module (adjust version as needed)

# Activate your Python environment (if needed)
source /home/ryanckh/miniconda3/etc/profile.d/conda.sh
conda activate vqa

# Run commands 
cd /shared_data/p_vidalr/ryanckh/GraphVQA
python eval_blip2.py
