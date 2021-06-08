#!/bin/bash
#SBATCH -J ASpot
#SBATCH -o log/%x.%3a.%A.out
#SBATCH -e log/%x.%3a.%A.err
#SBATCH --time=5-00:0:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH -N 1
#SBATCH --mem=90G
#SBATCH --exclude=gpu212-14
##SBATCH -A conf-iccv-2021.03.25-ghanembs


date

echo "Loading anaconda..."
# module purge
module load anaconda3
module load cuda/10.1.243
module list
source activate CALF-pytorch
echo "...Anaconda env loaded"


echo "Running python script..."
# This job has an ID:  $SLURM_ARRAY_TASK_ID
# echo $SLURM_ARRAY_TASK_ID 
python src/main.py \
--SoccerNet_path=/ibex/scratch/giancos/SoccerNet_calibration/ \
--features=ResNET_TF2_PCA512.npy \
--num_features=512 \
--model_name=CALF_subj_SBATCH \
--batch_size 32 \
--evaluation_frequency 20 \
--chunks_per_epoch 18000 \
"$@"

echo "... script terminated"


date
