import argparse
from d2l import torch as d2l
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader
from data_loader.data_loader import load_dataset_seq_seq_time_resource_fold_csv
from pathlib import Path
import torch
torch.manual_seed(42)

# Set up mandatory folders
Path("./results/models/").mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--num_folds", type=int, required=True)

args = parser.parse_args()
dataset = args.dataset
beam_width = 5
num_folds = 5 if args.num_folds is None else args.num_folds

#train_dataset_fold, val_dataset_fold, test_dataset_fold, dataset_config, num_activities_fold = load_dataset_seq_seq_time_fold_csv(dataset, num_folds)
train_dataset_fold, val_dataset_fold, test_dataset_fold, dataset_config, num_activities_fold, num_resources_fold = load_dataset_seq_seq_time_resource_fold_csv(dataset, num_folds)

max_length_trace = dataset_config['max_length_trace']
batch_size, num_steps = 128, max_length_trace
device = d2l.try_gpu()


tags = [i for i in range(num_steps)]
heights = [-0.36,-0.18,0,0.18,0.36]

for i in range(num_folds):
    train_iter = DataLoader(train_dataset_fold[i], batch_size)
    lengths_folds = [0]*num_steps
    for batch in train_iter:      
        X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
        for current_length in X_valid_len:
            lengths_folds[current_length-1] += 1
    plt.bar([pos+heights[i] for pos in tags[1:]],lengths_folds[1:],0.18,label="Fold " + str(i))

plt.xlabel("Prefix Length")
plt.ylabel("Frequency") 
plt.title("Prefix Length Frequency for " + str(dataset))
plt.xticks(tags[1:])
#plt.yscale("log")
plt.legend()
plt.savefig('./visualizations/hist_'+dataset+'.png',bbox_inches='tight',dpi=300)