#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH --time=72:00:00
PYTHON=/mnt/netapp2/Store_uni/home/usc/ci/jva/miniconda3/envs/abasp/bin/python3.9

NUM_FOLDS=5
for file in data/data_complete/*.csv
do
    name=${file##*/}
    base=${name%.csv}
    for script in "generate_histograms.py"
    do
        $PYTHON $script --dataset $base --num_folds $NUM_FOLDS
    done
done
