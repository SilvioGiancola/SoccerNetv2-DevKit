#!/bin/bash
#SBATCH --array=0-549
#SBATCH -J SN_feat
#SBATCH -o log/%x.%3a.%A.out
#SBATCH -e log/%x.%3a.%A.err
#SBATCH --time=0-03:59:00
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=45G

date

echo "Loading anaconda..."
# module purge
module load anaconda3
module load cuda/10.1.243
module list
source activate SoccerNet
echo "...Anaconda env loaded"


echo "Extracting features..."
python tools/ExtractResNET_TF2.py \
--soccernet_dirpath /ibex/scratch/giancos/SoccerNet/ \
--game_ID $SLURM_ARRAY_TASK_ID \
--back_end=TF2 \
--features=ResNET \
--video LQ \
--transform crop \
--verbose \
"$@"

echo "Features extracted..."

date
