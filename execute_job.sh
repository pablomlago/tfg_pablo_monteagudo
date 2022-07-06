#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
PYTHON=/mnt/netapp2/Store_uni/home/usc/ci/jva/miniconda3/envs/abasp/bin/python3.9

while [[ "$#" -gt 0 ]]; do case $1 in
    -d|--dataset) dataset="$2"; shift; shift ;;
    -e|--execution_id) execution_id="$2"; shift; shift ;;
    -n|--fold_num) fold_num="$2"; shift; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done


$PYTHON abasp.py --dataset $dataset --execution_id $execution_id --num_epochs 150 --num_folds 5 --fold_num $fold_num

