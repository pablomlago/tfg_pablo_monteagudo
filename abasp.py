import argparse
from d2l import torch as d2l
import pandas as pd

from torch.utils.data import DataLoader
from data_loader.data_loader import load_dataset_seq_seq_time_fold_csv, load_dataset_seq_seq_time_resource_fold_csv

from model.model import Seq2SeqAttentionDecoderNoPositional, Seq2SeqDecoder, Seq2SeqEncoderPositional, Seq2SeqAttentionDecoderPositional, Seq2SeqEncoderPositionalResource, Seq2SeqEncoderResourceNoPositional, Seq2SeqMultiHeadAttentionDecoderNoPositional
from predicter.predicter import batched_beam_decode, batched_beam_decode_optimized, predict_seq2seq
from trainer.trainer import train_seq2seq_mixed
from model.metric import levenshtein_similarity

from pathlib import Path
import torch
torch.manual_seed(42)

# Set up mandatory folders
Path("./results/models/").mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--execution_id", type=str, required=True)
parser.add_argument("--num_epochs", type=int, required=True)
parser.add_argument("--num_folds", type=int, required=True)
parser.add_argument("--fold_num", type=int, required=True)
parser.add_argument("--num_heads", type=int, default=1)
# With "action=store_true", we do not need an argument for the flag
parser.add_argument("--train", action="store_true")
# With choices we can select the type of postprocessing technique to apply
parser.add_argument("--postprocessing", type=str, required=True, choices=["beam", "beam_length_normalized", "beam_monteagudo", "argmax", "random"])
parser.add_argument("--disable_attention", required=False, action="store_true")
parser.add_argument("--optimize_beam_width", required=False, action="store_true")
parser.add_argument("--store_prefixes_in_results", required=False, action="store_true")

args = parser.parse_args()
dataset = args.dataset
execution_id = args.execution_id
num_epochs = args.num_epochs
num_heads = args.num_heads
train_required = args.train
postprocessing_type = args.postprocessing
beam_width = 5
num_folds = 5 if args.num_folds is None else args.num_folds

if not args.disable_attention:
    execution_name = f'results_attention/{dataset}_{args.execution_id}_{args.num_epochs}'
else:
    execution_name = f'results_no_attention/{dataset}_{args.execution_id}_{args.num_epochs}'

#train_dataset_fold, val_dataset_fold, test_dataset_fold, dataset_config, num_activities_fold = load_dataset_seq_seq_time_fold_csv(dataset, num_folds)
train_dataset_fold, val_dataset_fold, test_dataset_fold, dataset_config, num_activities_fold, num_resources_fold = load_dataset_seq_seq_time_resource_fold_csv(dataset, num_folds)

max_length_trace = dataset_config['max_length_trace']
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 128, max_length_trace
time_features = 6
lr, device = 0.005, d2l.try_gpu()

i = args.fold_num

num_activities = num_activities_fold[i]
# TODO: this is not very elegant, but it works
try:
    num_resources = num_resources_fold[i]
except IndexError:
    num_resources = -3
train_iter = DataLoader(train_dataset_fold[i], batch_size)
val_iter = DataLoader(val_dataset_fold[i], batch_size)
test_iter = DataLoader(test_dataset_fold[i], batch_size)

#encoder = Seq2SeqEncoderPositional(num_activities+3, embed_size, num_hiddens, time_features, num_layers,
#                        dropout)
encoder = Seq2SeqEncoderResourceNoPositional(num_activities+3, num_resources+3, embed_size, num_hiddens, time_features, num_layers,
                        dropout)
if not args.disable_attention:
    if num_heads == 1:
        decoder = Seq2SeqAttentionDecoderNoPositional(num_activities+3, embed_size, num_hiddens, num_layers,
                                dropout)
    else:
        decoder = Seq2SeqMultiHeadAttentionDecoderNoPositional(num_activities+3, embed_size, num_hiddens, num_layers, num_heads,
                                dropout)
else:
    decoder = Seq2SeqDecoder(num_activities+3, embed_size, num_hiddens, num_layers,
                             dropout)
net = d2l.EncoderDecoder(encoder, decoder)

name = dataset + "_fold_" + str(i)
if args.disable_attention:
    name = "disable_attention_" + name
if train_required:
    print("Training...")
    losses = train_seq2seq_mixed(execution_name, net, train_iter, val_iter, lr, num_epochs, device, name)
    result = pd.DataFrame(columns=['epoch','loss'])
    for epoch, loss in losses:
        result = result.append({'epoch': epoch, 'loss': loss[0]}, ignore_index=True)
    if args.disable_attention:
        result.to_csv(execution_name + '_disable_attention_fold_' + str(i) + '_losses.csv', index=False)
    else:
        result.to_csv(execution_name+'_fold_'+str(i)+'_losses.csv', index=False)


# If no postprocessing is enabled ABASP exits
if postprocessing_type is None:
    exit(0)

# If we do not train, we need to move the model to the gpu, otherwise we won't be able to load de weights properly
device = d2l.try_gpu()
net.to(device)

if args.optimize_beam_width:
    optimization_results = {}
    beam_width_to_try = [15, 10, 3, 5, 7, 10, 1]
    #beam_width_to_try = [3, 5]
    if "beam" in postprocessing_type:
        for j in beam_width_to_try:
            print("Optimizing beam width: " + str(j))
            if num_activities + 2 < j:
                print("Beam width " + str(j) + " too high, skipping")
                continue
            curr_beam_width = j
            predictions = batched_beam_decode_optimized(net, val_iter, num_steps, curr_beam_width, num_activities+2, device, name, not args.disable_attention, postprocessing_type=postprocessing_type)
            val_result = pd.DataFrame(columns=['prediction', 'truth', 'similarity'])
            for ((_,_,Y_ground_truth,_), predictions_batch) in zip(val_iter, predictions):
                for pred_trace, ground_truth_trace in zip(predictions_batch.tolist(), Y_ground_truth.tolist()):
                    l1, l2, similarity = levenshtein_similarity(pred_trace, ground_truth_trace, num_activities+2)
                    val_result = val_result.append({'prediction': pred_trace[:l1], 'truth': ground_truth_trace[:l2], 'similarity': similarity}, ignore_index=True)
            avg_val_dl = val_result['similarity'].mean()
            optimization_results[curr_beam_width] = avg_val_dl
        beam_width = max(optimization_results, key=optimization_results.get)
        print("Best beam width: " + str(beam_width))

        with open(execution_name + '_fold_' + str(i) + '_beam_width_optimization_results.csv', 'w') as f:
            for key, value in optimization_results.items():
                f.write("%s,%s\n" % (key, value))



print("Testing...")
if "beam" in postprocessing_type:
    predictions = batched_beam_decode_optimized(net, test_iter, num_steps, beam_width, num_activities+2, device, name, not args.disable_attention, postprocessing_type=postprocessing_type)
else:
    predictions = predict_seq2seq(net, test_iter, num_steps, device, name, not args.disable_attention, batch_size, postprocessing_strategy=postprocessing_type)


if not args.store_prefixes_in_results:
    result = pd.DataFrame(columns=['prediction','truth','similarity'])
    for ((_,_,Y_ground_truth,_), predictions_batch) in zip(test_iter, predictions):
        for pred_trace, ground_truth_trace in zip(predictions_batch.tolist(), Y_ground_truth.tolist()):
            l1, l2, similarity = levenshtein_similarity(pred_trace, ground_truth_trace, num_activities+2)
            result = result.append({'prediction': pred_trace[:l1], 'truth': ground_truth_trace[:l2], 'similarity': similarity}, ignore_index=True)
    print(result['similarity'].mean())
    if args.disable_attention:
        if args.optimize_beam_width:
            result.to_csv(execution_name + '_disable_attention_fold_' + str(i) + "_postprocessing_" + args.postprocessing + "_beam_width_" + str(beam_width) + '.csv', index=False)
        else:
            result.to_csv(execution_name + '_disable_attention_fold_' + str(i) + "_postprocessing_" + args.postprocessing + '.csv', index=False)
    else:
        if args.optimize_beam_width:
            result.to_csv(execution_name+'_fold_'+str(i)+"_postprocessing_"+args.postprocessing+ "_beam_width_" + str(beam_width) + '.csv', index=False)
        else:
            result.to_csv(execution_name+'_fold_'+str(i)+"_postprocessing_"+args.postprocessing+'.csv', index=False)

else:
    result = pd.DataFrame(columns=["prefix_length", 'prediction','truth','similarity'])
    for ((_,enc_len,Y_ground_truth,_), predictions_batch) in zip(test_iter, predictions):
        for prefix_length, pred_trace, ground_truth_trace in zip(enc_len, predictions_batch.tolist(), Y_ground_truth.tolist()):
            l1, l2, similarity = levenshtein_similarity(pred_trace, ground_truth_trace, num_activities+2)
            result = result.append({'prediction': pred_trace[:l1], 'truth': ground_truth_trace[:l2], 'similarity': similarity, "prefix_length" : prefix_length.numpy()}, ignore_index=True)
    print(result['similarity'].mean())
    if args.disable_attention:
        if args.optimize_beam_width:
            result.to_csv(execution_name + '_with_prefix_disable_attention_fold_' + str(i) + "_postprocessing_" + args.postprocessing + "_beam_width_" + str(beam_width) + '.csv', index=False)
        else:
            result.to_csv(execution_name + '_with_prefix_disable_attention_fold_' + str(i) + "_postprocessing_" + args.postprocessing + '.csv', index=False)
    else:
        if args.optimize_beam_width:
            result.to_csv(execution_name+'_with_prefix_fold_'+str(i)+"_postprocessing_"+args.postprocessing+ "_beam_width_" + str(beam_width) + '.csv', index=False)
        else:
            result.to_csv(execution_name+'_with_prefix_fold_'+str(i)+"_postprocessing_"+args.postprocessing+'.csv', index=False)


