#!/bin/bash
EXECUTION_ID="experimentation_tfg"
NUM_FOLDS=5

for file in data/data_complete/*.csv
do
    name=${file##*/}
    base=${name%.csv}
    for script in "abasp.py"
    do
      for i in {0..4}; do
	      sbatch ./execute_job.sh --dataset $base --execution_id $EXECUTION_ID --fold_num $i
	    done
    done
done
