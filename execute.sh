#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
EXECUTION_ID="experimentation_tfg"
NUM_FOLDS=5
for file in data/data_complete/*.csv
do
    name=${file##*/}
    base=${name%.csv}
    for script in "abasp.py"
    do
        python3 $script --dataset $base --execution_id $EXECUTION_ID --num_epochs 100 --num_folds $NUM_FOLDS
    done
done
