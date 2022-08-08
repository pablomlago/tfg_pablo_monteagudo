#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH --time=72:00:00
PYTHON=/mnt/netapp2/Store_uni/home/usc/ci/jva/miniconda3/envs/abasp/bin/python3.9

while [[ "$#" -gt 0 ]]; do case $1 in
    -d|--dataset) dataset="$2"; shift; shift ;;
    -e|--execution_id) execution_id="$2"; shift; shift ;;
    -n|--fold_num) fold_num="$2"; shift; shift ;;
    -p|--postprocessing) postprocessing="$2"; shift; shift ;;
    -a|--disable_attention) disable_attention="$1";  shift ;;
    -o|--optimize_beam_width) optimize_beam_width="$1"; shift ;;
    -s|--store_prefixes_in_results) store_prefixes_in_results="$1"; shift ;;
    -t|--train) train="$1"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

command="$PYTHON abasp.py --dataset $dataset --execution_id $execution_id --num_epochs 150 --num_folds 5 --fold_num $fold_num --postprocessing $postprocessing"
if [ "$disable_attention" == "--disable_attention" ]; then
  command="$command --disable_attention"
fi
if [ "$optimize_beam_width" == "--optimize_beam_width" ]; then
  command="$command --optimize_beam_width"
fi
if [ "$train" == "--train" ]; then
  command="$command --train"
fi
if [ "$store_prefixes_in_results" == "--store_prefixes_in_results" ]; then
  command="$command --store_prefixes_in_results"
fi
echo $command
$command

