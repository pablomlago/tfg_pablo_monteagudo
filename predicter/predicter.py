from queue import PriorityQueue

from numpy import dtype
from model.metric import levenshtein_similarity
from math import log
import torch

class BeamSearchNode(object):
    def __init__(self, dec_state, previous_node, dec_X, log_prob, length, eot_found):
        self.dec_state = dec_state
        self.previous_node = previous_node
        self.dec_X = dec_X
        self.log_prob = log_prob
        self.length = length
        self.eot_found = eot_found

    def eval(self, alpha=0.2):
        reward = 0
        return self.log_prob
        #return self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.length < other.length

    def __le__(self,other):
        return self.length < other.length

def batched_beam_decode(net, data_iter, num_steps, beam_size, eot_token,
                    device, name, save_attention_weights=False):

    """Predict for sequence to sequence."""
    # We load the best model parameters
    net.load_state_dict(torch.load(name + '_best-model-parameters.pt'))
    # Set `net` to eval mode for inference
    net.eval()
    #We get predictions for each batch
    preds = []
    for batch in data_iter:
        batch_pred = []
        batched_enc_X, batched_enc_valid_len, _, _ = [x.to(device) for x in batch]
        for (enc_X, enc_valid_len) in zip(batched_enc_X, batched_enc_valid_len):
            enc_X = enc_X.unsqueeze(0)
            enc_valid_len = enc_valid_len.unsqueeze(0)
            #We get the outputs of the encoder
            enc_outputs = net.encoder(enc_X, enc_valid_len)
            dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
            #We prepare the first output for the decoder
            dec_X = torch.unsqueeze(enc_X[:,-1,0],1)
            #Starting node
            initial_node = BeamSearchNode(dec_state, None, dec_X, 0, 0, False)

            #Decode for one step using decoder
            Y, dec_state = net.decoder(dec_X, dec_state)
            #Beam search
            values, idxs = Y.softmax(dim=2).topk(beam_size, dim=2)

            #Open nodes
            open_nodes = PriorityQueue()

            for j in range(beam_size):          
                current_idx = idxs[0][0][j].view(1, -1)
                current_value = values[0][0][j].item()
                new_length = initial_node.length + (1 if not initial_node.eot_found and current_idx != eot_token else 0)

                node = BeamSearchNode(dec_state, initial_node, current_idx, 
                    initial_node.log_prob + log(current_value), new_length,  initial_node.eot_found or current_idx == eot_token)

                score = -node.eval()
                open_nodes.put((score, node))

            for _ in range(1,num_steps):
                new_nodes = PriorityQueue()
                for j in range(beam_size):
                    _, current_node = open_nodes.get()
                    dec_X = current_node.dec_X
                    dec_state = current_node.dec_state

                    #Decode for one step using decoder
                    Y, dec_state = net.decoder(dec_X, dec_state)
                    #Beam search
                    values, idxs = Y.softmax(dim=2).topk(beam_size, dim=2)

                    for j in range(beam_size):        
                        current_idx = idxs[0][0][j].view(1, -1)
                        current_value = values[0][0][j].item()
                        new_length = current_node.length + (1 if not current_node.eot_found and current_idx != eot_token else 0)
                        node = BeamSearchNode(dec_state, current_node, current_idx, 
                            current_node.log_prob + log(current_value), new_length, current_node.eot_found or current_idx == eot_token)
                        score = -node.eval()
                        new_nodes.put((score, node))
                for j in range(beam_size):
                    open_nodes.put(new_nodes.get())
            #We prepare an aditional variable for storing the results
            current_pred = []
            _, best_node = open_nodes.get()
            for i in range(num_steps-1,-1,-1):
                current_pred = [best_node.dec_X.item()] + current_pred
                best_node = best_node.previous_node
            batch_pred.append(current_pred)
        preds.append(torch.Tensor(batch_pred).to(torch.int))
    return preds

def batched_beam_decode_optimized(net, data_iter, num_steps, beam_size, eot_token,
                    device, name, save_attention_weights=False):

    """Predict for sequence to sequence."""
    # We load the best model parameters
    net.load_state_dict(torch.load(name + '_best-model-parameters.pt'))
    # Set `net` to eval mode for inference
    net.eval()
    #We get predictions for each batch
    preds = []
    for batch in data_iter:
        batch_pred = []
        batched_enc_X, batched_enc_valid_len, _, _ = [x.to(device) for x in batch]
        for (enc_X, enc_valid_len) in zip(batched_enc_X, batched_enc_valid_len):
            enc_X = enc_X.unsqueeze(0)
            enc_valid_len = enc_valid_len.unsqueeze(0)
            #We get the outputs of the encoder
            enc_outputs = net.encoder(enc_X, enc_valid_len)
            dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
            #We prepare the first output for the decoder
            dec_X = torch.unsqueeze(enc_X[:,-1,0],1)
            #Starting node
            initial_node = BeamSearchNode(dec_state, None, dec_X, 0, 0, False)

            #Decode for one step using decoder
            Y, dec_state = net.decoder(dec_X, dec_state)
            #Beam search
            values, idxs = Y.softmax(dim=2).topk(beam_size, dim=2)

            #Open nodes
            open_nodes = PriorityQueue()

            for j in range(beam_size):          
                current_idx = idxs[0][0][j].view(1, -1)
                current_value = values[0][0][j].item()
                new_length = initial_node.length + (1 if not initial_node.eot_found and current_idx != eot_token else 0)
                node = BeamSearchNode(dec_state, initial_node, current_idx, 
                    initial_node.log_prob + log(current_value), new_length,  initial_node.eot_found or current_idx == eot_token)
                score = -node.eval()
                open_nodes.put((score, node))

            for _ in range(1,num_steps):
                new_nodes = PriorityQueue()
                for j in range(beam_size):
                    current_score, current_node = open_nodes.get()

                    if current_node.eot_found:
                        new_nodes.put((current_score, current_node))
                    else:
                        dec_X = current_node.dec_X
                        dec_state = current_node.dec_state

                        #Decode for one step using decoder
                        Y, dec_state = net.decoder(dec_X, dec_state)
                        #Beam search
                        values, idxs = Y.softmax(dim=2).topk(beam_size, dim=2)

                        for j in range(beam_size):        
                            current_idx = idxs[0][0][j].view(1, -1)
                            current_value = values[0][0][j].item()
                            new_length = current_node.length + (1 if not current_node.eot_found and current_idx != eot_token else 0)
                            node = BeamSearchNode(dec_state, current_node, current_idx, 
                                current_node.log_prob + log(current_value), new_length, current_node.eot_found or current_idx == eot_token)
                            score = -node.eval()
                            new_nodes.put((score, node))
                for j in range(beam_size):
                    open_nodes.put(new_nodes.get())
            #We prepare an aditional variable for storing the results
            current_pred = []
            _, best_node = open_nodes.get()
            #for i in range(num_steps-1,-1,-1):
            #    current_pred = [best_node.dec_X.item()] + current_pred
            #    best_node = best_node.previous_node
            while best_node is not None:
                current_pred = [best_node.dec_X.item()] + current_pred
                best_node = best_node.previous_node
            batch_pred.append(current_pred+[eot_token]*(num_steps-len(current_pred)))
        preds.append(torch.Tensor(batch_pred).to(torch.int))
    return preds