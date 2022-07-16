import argparse
from d2l import torch as d2l
import pandas as pd

from torch.utils.data import DataLoader
from data_loader.data_loader import load_dataset_seq_seq_time_fold_csv, load_dataset_seq_seq_time_resource_fold_csv

from model.model import Seq2SeqAttentionDecoderNoPositional, Seq2SeqDecoder, Seq2SeqEncoderPositional, Seq2SeqAttentionDecoderPositional, Seq2SeqEncoderPositionalResource, Seq2SeqEncoderResourceNoPositional
from predicter.predicter import batched_beam_decode, batched_beam_decode_optimized, predict_seq2seq
from trainer.trainer import train_seq2seq_mixed
from model.metric import levenshtein_similarity

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--execution_id", type=str, required=True)
parser.add_argument("--num_epochs", type=int, required=True)
parser.add_argument("--num_folds", type=int, required=True)
parser.add_argument("--fold_num", type=int, required=True)
parser.add_argument("--train", type=bool, required=False, default=True)
parser.add_argument("--beam", type=bool, required=False, default=True)

args = parser.parse_args()
dataset = args.dataset
execution_id = args.execution_id
num_epochs = args.num_epochs
train_required = args.train
beam_enabled = args.beam
beam_width = 5
num_folds = 5 if args.num_folds is None else args.num_folds

execution_name = f'results/{dataset}_{args.execution_id}_{args.num_epochs}'

#train_dataset_fold, val_dataset_fold, test_dataset_fold, dataset_config, num_activities_fold = load_dataset_seq_seq_time_fold_csv(dataset, num_folds)
train_dataset_fold, val_dataset_fold, test_dataset_fold, dataset_config, num_activities_fold, num_resources_fold = load_dataset_seq_seq_time_resource_fold_csv(dataset, num_folds)

max_length_trace = dataset_config['max_length_trace']
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 128, max_length_trace
time_features = 6
lr, device = 0.005, d2l.try_gpu()

i = args.fold_num

num_activities = num_activities_fold[i]
num_resources = num_resources_fold[i]
train_iter = DataLoader(train_dataset_fold[i], batch_size)
val_iter = DataLoader(train_dataset_fold[i], batch_size)
test_iter = DataLoader(test_dataset_fold[i], batch_size)

#encoder = Seq2SeqEncoderPositional(num_activities+3, embed_size, num_hiddens, time_features, num_layers,
#                        dropout)
encoder = Seq2SeqEncoderResourceNoPositional(num_activities+3, num_resources+3, embed_size, num_hiddens, time_features, num_layers,
                        dropout)
decoder = Seq2SeqDecoder(num_activities+3, embed_size, num_hiddens, num_layers,
                        dropout)
net = d2l.EncoderDecoder(encoder, decoder)

if train_required:
    losses = train_seq2seq_mixed(execution_name, net, train_iter, val_iter, lr, num_epochs, device, dataset + "_fold_" + str(i))
    result = pd.DataFrame(columns=['epoch','loss'])
    for epoch, loss in losses:
        result = result.append({'epoch': epoch, 'loss': loss[0]}, ignore_index=True)
    result.to_csv(execution_name+'_fold_'+str(i)+'_losses.csv', index=False)

if beam_enabled:
    predictions = batched_beam_decode_optimized(net, test_iter, num_steps, beam_width, num_activities+2, device, dataset + "_fold_" + str(i))
else:
    predictions = predict_seq2seq(net, test_iter, num_steps, device, dataset + "_fold_" + str(i))


result = pd.DataFrame(columns=['prediction','truth','similarity'])
for ((_,_,Y_ground_truth,_), predictions_batch) in zip(test_iter, predictions):
    for pred_trace, ground_truth_trace in zip(predictions_batch.tolist(), Y_ground_truth.tolist()):
        l1, l2, similarity = levenshtein_similarity(pred_trace, ground_truth_trace, num_activities+2)
        result = result.append({'prediction': pred_trace[:l1], 'truth': ground_truth_trace[:l2], 'similarity': similarity}, ignore_index=True)
print(result['similarity'].mean())
result.to_csv(execution_name+'_fold_'+str(i)+'.csv', index=False)

