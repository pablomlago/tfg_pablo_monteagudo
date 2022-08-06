import argparse
from d2l import torch as d2l
import pandas as pd
import pickle

from pathlib import Path
import torch

import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np

import command

# Set up mandatory folders
Path("./results/models/").mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--execution_id", type=str, required=True)
parser.add_argument("--num_epochs", type=int, required=True)
parser.add_argument("--num_folds", type=int, required=True)
parser.add_argument("--fold_num", type=int, required=True)
parser.add_argument("--num_heads", type=int, default=1)
parser.add_argument("--idx", type=int, default=0)
# With choices we can select the type of postprocessing technique to apply
parser.add_argument("--postprocessing", type=str, required=True, choices=["beam", "beam_length_normalized", "beam_monteagudo", "argmax", "random"])

args = parser.parse_args()
dataset = args.dataset
execution_id = args.execution_id
num_epochs = args.num_epochs
num_heads = args.num_heads
postprocessing_type = args.postprocessing
beam_width = 5
num_folds = 5 if args.num_folds is None else args.num_folds
i = args.fold_num
idx = args.idx

execution_name = f'results_attention/{dataset}_{args.execution_id}_{args.num_epochs}'

attention_weights = pickle.load(open(execution_name+'_fold_'+str(i)+"_attention_weights", 'rb'))
max_length_trace = len(attention_weights[0][0])

epsilon = 0.000001

prefix_lengths = list(map(lambda x : len([index for index,value in enumerate(x[0]) if value > epsilon]),attention_weights))
fixed_length = 4
attention_weights_fixed_length = [attention_weights[i] for i in range(len(attention_weights)) if prefix_lengths[i] == fixed_length]

dict_length = {}
for j in range(fixed_length):
    dict_length[j] = []
for attention_weight in attention_weights_fixed_length:
    attention_weight = list(map(list, zip(*attention_weight)))
    for j in range(fixed_length):
        dict_length[j].append(np.std(attention_weight[max_length_trace+j-fixed_length]))
print(dict_length)

fig, ax = plt.subplots()
ax.boxplot(dict_length.values())
ax.set_xticklabels(dict_length.keys())
plt.show()

 
result = pd.read_csv(execution_name+"_fold_"+str(i)+"_postprocessing_" + args.postprocessing + ".csv")

ax = sns.heatmap(attention_weights[idx], annot=True)
plt.xlabel('''Attention Scoring

Prediction: {}, Truth: {}, DL: {}'''.format(result.iloc[idx, :]["prediction"],result.iloc[idx, :]["truth"], result.iloc[idx, :]["similarity"]))
plt.ylabel("Suffix Sequence Position")
ax.text(.5, .05, "test", ha='center')
plt.show()


