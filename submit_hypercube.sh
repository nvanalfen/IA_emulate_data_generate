#!/bin/bash
#SBATCH -J ia_ml
#SBATCH --partition=short
#SBATCH --time=22:00:00
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --array=1-50%50
#SBATCH --output=output_logs/%A-%a.out
#SBATCH --error=error_logs/%A-%a.err
#SBATCH --mail-user=nvanalfen2@gmail.com
#SBATCH --mail-type=ALL

#module load discovery anaconda3/3.7
source /home/nvanalfen/miniconda3/bin/activate
conda activate ia_ml

JOB=$(($SLURM_ARRAY_TASK_ID*1))

python generate_training_data_hypercube.py $JOB 200 bolplanck_hypercube.npz

