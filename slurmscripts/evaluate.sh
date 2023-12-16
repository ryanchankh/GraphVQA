#!/bin/bash
#SBATCH --partition=gpu           # Specify the partition or queue name
#SBATCH --nodes=1
#SBATCH --mem=150G                # Memory to allocate (in GB) #SBATCH
#SBATCH --output=/shared_data/p_vidalr/ryanckh/GraphVQA/slurmlog/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --time=96:00:00                 # Maximum runtime (format: HH:MM:SS)

# Load any necessary modules
module load cuda/11.8           # Load Anaconda module (adjust version as needed)

# Activate your Python environment (if needed)
conda init bash
conda activate vqa

# Run commands 
cd /shared_data/p_vidalr/ryanckh/GraphVQA
python pipeline_model_gat.py 
python gqa_dataset_entry.py 
