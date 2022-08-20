import math
import operator
from queue import PriorityQueue

from pexpect import EOF
from model.metric import levenshtein_similarity
from math import log
import torch
import os


class BeamSearchNode(object):
    def __init__(self, dec_state, previous_node, dec_X, log_prob, length, eot_found):
        self.dec_state = dec_state
        self.previous_node = previous_node
        self.dec_X = dec_X
        self.log_prob = log_prob
        self.length = length
        self.eot_found = eot_found

    def eval(self, alpha=0.75):
        reward = 0
        return self.log_prob
        # return self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.length < other.length

    def __le__(self, other):
        return self.length < other.length


def batched_beam_decode(net, data_iter, num_steps, beam_size, eot_token,
                        device, name, save_attention_weights=False):
    """Predict for sequence to sequence."""
    # We load the best model parameters
    net.load_state_dict(torch.load(name + '_best-model-parameters.pt'))
    # Set `net` to eval mode for inference
    net.eval()
    # We get predictions for each batch
    preds = []
    for batch in data_iter:
        batch_pred = []
        batched_enc_X, batched_enc_valid_len, _, _ = [x.to(device) for x in batch]
        for (enc_X, enc_valid_len) in zip(batched_enc_X, batched_enc_valid_len):
            enc_X = enc_X.unsqueeze(0)
            enc_valid_len = enc_valid_len.unsqueeze(0)
            # We get the outputs of the encoder
            enc_outputs = net.encoder(enc_X, enc_valid_len)
            dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
            # We prepare the first output for the decoder
            dec_X = torch.unsqueeze(enc_X[:, -1, 0], 1)
            # Starting node
            initial_node = BeamSearchNode(dec_state, None, dec_X, 0, 0, False)

            # Decode for one step using decoder
            Y, dec_state = net.decoder(dec_X, dec_state)
            # Beam search
            values, idxs = Y.softmax(dim=2).topk(beam_size, dim=2)

            # Open nodes
            open_nodes = PriorityQueue()

            for j in range(beam_size):
                current_idx = idxs[0][0][j].view(1, -1)
                current_value = values[0][0][j].item()
                new_length = initial_node.length + (1 if not initial_node.eot_found and current_idx != eot_token else 0)

                node = BeamSearchNode(dec_state, initial_node, current_idx,
                                      initial_node.log_prob + log(current_value), new_length,
                                      initial_node.eot_found or current_idx == eot_token)

                score = -node.eval()
                open_nodes.put((score, node))

            for _ in range(1, num_steps):
                new_nodes = PriorityQueue()
                for j in range(beam_size):
                    _, current_node = open_nodes.get()
                    dec_X = current_node.dec_X
                    dec_state = current_node.dec_state

                    # Decode for one step using decoder
                    Y, dec_state = net.decoder(dec_X, dec_state)
                    # Beam search
                    values, idxs = Y.softmax(dim=2).topk(beam_size, dim=2)

                    for j in range(beam_size):
                        current_idx = idxs[0][0][j].view(1, -1)
                        current_value = values[0][0][j].item()
                        new_length = current_node.length + (
                            1 if not current_node.eot_found and current_idx != eot_token else 0)
                        node = BeamSearchNode(dec_state, current_node, current_idx,
                                              current_node.log_prob + log(current_value), new_length,
                                              current_node.eot_found or current_idx == eot_token)
                        score = -node.eval()
                        new_nodes.put((score, node))
                for j in range(beam_size):
                    open_nodes.put(new_nodes.get())
            # We prepare an aditional variable for storing the results
            current_pred = []
            _, best_node = open_nodes.get()
            for i in range(num_steps - 1, -1, -1):
                current_pred = [best_node.dec_X.item()] + current_pred
                best_node = best_node.previous_node
            batch_pred.append(current_pred)
        preds.append(torch.Tensor(batch_pred).to(torch.int))
    return preds


class BeamSearchNodeOptimized(object):
    def __init__(self, dec_state, previous_node, dec_X, log_prob, length, eot_found, coverage_weights):
        self.dec_state = dec_state
        self.previous_node = previous_node
        self.dec_X = dec_X
        self.log_prob = log_prob
        self.length = length
        self.eot_found = eot_found
        self.coverage_weights = coverage_weights

    def eval(self, beam_type, alpha=0.65, beta=0.65):
        reward = 0
        if beam_type == "beam":
            return self.log_prob
        elif beam_type == "beam_length_normalized":
            # Following https://arxiv.org/pdf/1609.08144.pdf is not a product but a division (log(P(Y|X))/lp(Y))
            return self.log_prob / (math.pow(5 + self.length, alpha) / math.pow(5 + 1, alpha))
        elif beam_type == "beam_length_normalized_coverage":
            # Following https://arxiv.org/pdf/1609.08144.pdf is not a product but a division (log(P(Y|X))/lp(Y))
            return self.log_prob / (math.pow(5 + self.length, alpha) / math.pow(5 + 1, alpha)) + beta*sum([log(weight) for weight in self.coverage_weights])
        elif beam_type == "beam_monteagudo":
            return self.log_prob * (math.pow(5 + self.length, alpha) / math.pow(5 + 1, alpha))
        else:
            raise ValueError("Unknown beam type")

    def __lt__(self, other):
        return self.length < other.length

    def __le__(self, other):
        return self.length < other.length

def setup_reproducibility():
    # Set seeds for reproducibility
    # Call this function just before predicting. Otherwise, the predictions will be different if we train
    # and test and if we only test using the saved weights
    import torch
    import random
    import numpy as np
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)


def batched_beam_decode_optimized(net, data_iter, num_steps, beam_size, eot_token,
                                  device, name, attention_enabled, postprocessing_type, save_attention_weights=False):
    setup_reproducibility()
    """Predict for sequence to sequence."""
    # We load the best model parameters
    if attention_enabled:
        net.load_state_dict(torch.load(os.path.join("./results_attention/models", name + '_best-model-parameters.pt')))
    else:
        net.load_state_dict(
            torch.load(os.path.join("./results_no_attention/models", name + '_best-model-parameters.pt')))
    # Set `net` to eval mode for inference
    net.eval()
    # We get predictions for each batch
    preds = []
    for batch in data_iter:
        batch_pred = []
        batched_enc_X, batched_enc_valid_len, _, _ = [x.to(device) for x in batch]
        for (enc_X, enc_valid_len) in zip(batched_enc_X, batched_enc_valid_len):
            enc_X = enc_X.unsqueeze(0)
            enc_valid_len = enc_valid_len.unsqueeze(0)
            # We get the outputs of the encoder
            enc_outputs = net.encoder(enc_X, enc_valid_len)
            dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
            # We prepare the first output for the decoder
            dec_X = torch.unsqueeze(enc_X[:, -1, 0], 1)
            # Starting node
            initial_node = BeamSearchNodeOptimized(dec_state, None, dec_X, 0, 0, False)

            open_nodes = PriorityQueue()
            end_nodes = []
            number_required = 1

            open_nodes.put((-initial_node.eval(postprocessing_type), initial_node))

            for _ in range(num_steps):
                current_score, current_node = open_nodes.get()
                dec_X = current_node.dec_X
                dec_state = current_node.dec_state

                if dec_X.item() == eot_token and current_node.previous_node != None:
                    end_nodes.append((current_score, current_node))
                    if len(end_nodes) >= number_required:
                        break
                    else:
                        continue

                # Decode for one step using decoder
                Y, dec_state = net.decoder(dec_X, dec_state)
                # Beam search
                values, idxs = Y.softmax(dim=2).topk(beam_size, dim=2)

                for j in range(beam_size):
                    current_idx = idxs[0][0][j].view(1, -1)
                    current_value = values[0][0][j].item()
                    new_length = current_node.length + (
                        1 if not current_node.eot_found and current_idx != eot_token else 0)
                    if attention_enabled:
                        node = BeamSearchNodeOptimized(dec_state, current_node, current_idx,
                                                    current_node.log_prob + log(current_value), new_length,
                                                    current_node.eot_found or current_idx == eot_token, None)
                    else:
                        node = BeamSearchNodeOptimized(dec_state, current_node, current_idx,
                                                    current_node.log_prob + log(current_value), new_length,
                                                    current_node.eot_found or current_idx == eot_token, None)
                    score = -node.eval(postprocessing_type)
                    open_nodes.put((score, node))

            if (len(end_nodes) == 0):
                end_nodes = [open_nodes.get() for _ in range(number_required)]
            # We prepare an aditional variable for storing the results
            current_pred = []
            _, best_node = sorted(end_nodes, key=operator.itemgetter(0))[0]
            while best_node.previous_node is not None:
                current_pred = [best_node.dec_X.item()] + current_pred
                best_node = best_node.previous_node
            batch_pred.append(current_pred + [eot_token] * (num_steps - len(current_pred)))
        preds.append(torch.Tensor(batch_pred).to(torch.int))
    return preds

def batched_beam_decode_optimized_prepadding(net, data_iter, num_steps, beam_size, eot_token,
                                  device, name, attention_enabled, postprocessing_type, save_attention_weights=False):
    setup_reproducibility()
    """Predict for sequence to sequence."""
    # We load the best model parameters
    if attention_enabled:
        net.load_state_dict(torch.load(os.path.join("./results_attention/models", name + '_best-model-parameters.pt')))
    else:
        net.load_state_dict(
            torch.load(os.path.join("./results_no_attention/models", name + '_best-model-parameters.pt')))
    # Set `net` to eval mode for inference
    net.eval()
    # We get predictions for each batch
    preds = []
    #Small-value for initialization
    epsilon = 0.0001
    for batch in data_iter:
        batch_pred = []
        batched_enc_X, batched_enc_valid_len, _, _ = [x.to(device) for x in batch]
        for (enc_X, enc_valid_len) in zip(batched_enc_X, batched_enc_valid_len):
            enc_X = enc_X.unsqueeze(0)
            enc_valid_len = enc_valid_len.unsqueeze(0)
            # We get the outputs of the encoder
            enc_outputs = net.encoder(enc_X, enc_valid_len)
            dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
            # We prepare the first output for the decoder
            dec_X = enc_X[:,:,0].gather(1,enc_valid_len.view(-1,1)-1)
            # Starting node
            initial_node = BeamSearchNodeOptimized(dec_state, None, dec_X, 0, 0, False, [epsilon]*enc_valid_len)

            open_nodes = PriorityQueue()
            end_nodes = []
            number_required = 1

            open_nodes.put((-initial_node.eval(postprocessing_type), initial_node))

            for _ in range(num_steps):
                current_score, current_node = open_nodes.get()
                dec_X = current_node.dec_X
                dec_state = current_node.dec_state

                if dec_X.item() == eot_token and current_node.previous_node != None:
                    end_nodes.append((current_score, current_node))
                    if len(end_nodes) >= number_required:
                        break
                    else:
                        continue

                # Decode for one step using decoder
                Y, dec_state = net.decoder(dec_X, dec_state)
                # Beam search
                values, idxs = Y.softmax(dim=2).topk(beam_size, dim=2)

                for j in range(beam_size):
                    current_idx = idxs[0][0][j].view(1, -1)
                    current_value = values[0][0][j].item()
                    new_length = current_node.length + (
                        1 if not current_node.eot_found and current_idx != eot_token else 0)
                    if attention_enabled:
                        attention_weights = net.decoder.attention_weights[0].squeeze()
                        coverage_weights = current_node.coverage_weights
                        coverage_weights = [min(1.0, coverage_weights[i]+attention_weights[i])for i in range(enc_valid_len)]
                        node = BeamSearchNodeOptimized(dec_state, current_node, current_idx,
                                                    current_node.log_prob + log(current_value), new_length,
                                                    current_node.eot_found or current_idx == eot_token, coverage_weights)
                    else:
                        node = BeamSearchNodeOptimized(dec_state, current_node, current_idx,
                                                    current_node.log_prob + log(current_value), new_length,
                                                    current_node.eot_found or current_idx == eot_token, None)
                    score = -node.eval(postprocessing_type)
                    open_nodes.put((score, node))

            if (len(end_nodes) == 0):
                end_nodes = [open_nodes.get() for _ in range(number_required)]
            # We prepare an aditional variable for storing the results
            current_pred = []
            _, best_node = sorted(end_nodes, key=operator.itemgetter(0))[0]
            while best_node.previous_node is not None:
                current_pred = [best_node.dec_X.item()] + current_pred
                best_node = best_node.previous_node
            batch_pred.append(current_pred + [eot_token] * (num_steps - len(current_pred)))
        preds.append(torch.Tensor(batch_pred).to(torch.int))
    return preds


def predict_seq2seq(net, data_iter, num_steps,
                    device, name, attention_enabled, batch_size, postprocessing_strategy, save_attention_weights=False):
    setup_reproducibility()
    """Predict for sequence to sequence."""
    # We load the best model parameters
    if attention_enabled:
        net.load_state_dict(torch.load(os.path.join("./results_attention/models", name + '_best-model-parameters.pt')))
    else:
        net.load_state_dict(
            torch.load(os.path.join("./results_no_attention/models", name + '_best-model-parameters.pt')))
    # Set `net` to eval mode for inference
    net.eval()
    # We get predictions for each batch
    preds = []
    final_preds = []
    for batch in data_iter:
        batch_enc_X, batch_enc_valid_len, _, _ = [x.to(device) for x in batch]
        for enc_X, enc_valid_len in zip(torch.unbind(batch_enc_X), torch.unbind(batch_enc_valid_len)):
            enc_X = enc_X.unsqueeze(0)
            enc_valid_len = enc_valid_len.unsqueeze(0)
            #print("Enc X: ", enc_X.shape)
            #print("Enc valid len: ", enc_valid_len.shape)
            # We get the outputs of the encoder
            enc_outputs = net.encoder(enc_X, enc_valid_len)
            dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
            # We prepare the first output for the decoder
            dec_X = enc_X[:, -1, 0].reshape(-1, 1).expand(-1, num_steps + 1).detach().clone()
            # We iterate over the steps in the decoder
            for i in range(num_steps):
                Y, dec_state = net.decoder(dec_X[:, :num_steps], dec_state)
                if postprocessing_strategy == "argmax":
                    dec_X[:, i + 1] = Y.argmax(dim=2)[:, i]
                elif postprocessing_strategy == "random":
                    dec_X[:, i + 1] = torch.multinomial(Y.softmax(dim=2)[:, i], 1)[:, 0]
                else:
                    raise ValueError("Unknown postprocessing strategy")
            preds.append(dec_X[:, 1:].to(torch.int).cpu())

    # We need to compact the predictions in chucks of the size of the batch in order to calculate
    # the similarity correctly.
    for n_batch, batch in enumerate(data_iter):
        my_arr = []
        for i in range(len(batch)):
            my_arr.append(preds[n_batch * batch_size + i])
        final_preds.append(torch.cat(my_arr, dim=0))

    return final_preds

def predict_seq2seq_optimized(net, data_iter, num_steps,
                    device, name, attention_enabled, batch_size, postprocessing_strategy, save_attention_weights=False):
    setup_reproducibility()
    """Predict for sequence to sequence."""
    # We load the best model parameters
    if attention_enabled:
        net.load_state_dict(torch.load(os.path.join("./results_attention/models", name + '_best-model-parameters.pt')))
    else:
        net.load_state_dict(
            torch.load(os.path.join("./results_no_attention/models", name + '_best-model-parameters.pt')))
    # Set `net` to eval mode for inference
    net.eval()
    #We get predictions for each batch
    preds = []
    for batch in data_iter:
        enc_X, enc_valid_len, _, _ = [x.to(device) for x in batch]
        #We get the outputs of the encoder
        enc_outputs = net.encoder(enc_X, enc_valid_len)
        dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
        #We prepare the first output for the decoder
        dec_X = torch.unsqueeze(enc_X[:,-1,0],1)
        #We prepare an aditional variable for storing the results
        pred = torch.empty((enc_X.shape[0],num_steps+1), dtype=torch.int)
        #We iterate over the steps in the decoder
        for i in range(num_steps):
            Y, dec_state = net.decoder(dec_X, dec_state)
            if postprocessing_strategy == "argmax":
                dec_X = Y.argmax(dim=2)
            elif postprocessing_strategy == "random":
                dec_X = torch.multinomial(Y.softmax(dim=2).squeeze(1), 1)
            else:
                raise ValueError("Unknown postprocessing strategy")
            #dec_X = Y.argmax(dim=2)
            pred[:,i+1] = dec_X.squeeze()
        preds.append(pred[:,1:])
    return preds

def predict_seq2seq_optimized_prepadding(net, data_iter, num_steps,
                    device, name, attention_enabled, batch_size, postprocessing_strategy, save_attention_weights=False):
    setup_reproducibility()
    """Predict for sequence to sequence."""
    # We load the best model parameters
    if attention_enabled:
        net.load_state_dict(torch.load(os.path.join("./results_attention/models", name + '_best-model-parameters.pt')))
    else:
        net.load_state_dict(
            torch.load(os.path.join("./results_no_attention/models", name + '_best-model-parameters.pt')))
    # Set `net` to eval mode for inference
    net.eval()
    #We get predictions for each batch
    preds = []
    for batch in data_iter:
        enc_X, enc_valid_len, _, _ = [x.to(device) for x in batch]
        #We get the outputs of the encoder
        enc_outputs = net.encoder(enc_X, enc_valid_len)
        dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
        #We prepare the first output for the decoder
        dec_X = enc_X[:,:,0].gather(1,enc_valid_len.view(-1,1)-1)
        #We prepare an aditional variable for storing the results
        pred = torch.empty((enc_X.shape[0],num_steps+1), dtype=torch.int)
        #We iterate over the steps in the decoder
        for i in range(num_steps):
            Y, dec_state = net.decoder(dec_X, dec_state)
            if postprocessing_strategy == "argmax":
                dec_X = Y.argmax(dim=2)
            elif postprocessing_strategy == "random":
                dec_X = torch.multinomial(Y.softmax(dim=2).squeeze(1), 1)
            else:
                raise ValueError("Unknown postprocessing strategy")
            #dec_X = Y.argmax(dim=2)
            pred[:,i+1] = dec_X.squeeze()
        preds.append(pred[:,1:])
    return preds