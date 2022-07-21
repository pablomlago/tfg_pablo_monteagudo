import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

from model.loss import SoftmaxCELoss

import os
def train_seq2seq_mixed(execution_name, net, train_iter, val_iter, lr, num_epochs, device, name):
    """Train a model for sequence to sequence."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = SoftmaxCELoss()
    losses = []
    best_model_loss = float('inf')
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        #We train the model
        net.train()
        for batch in train_iter:
            with torch.cuda.amp.autocast():
                optimizer.zero_grad()         
                X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
                dec_input = torch.cat([X[:,-1,0].reshape(-1, 1), Y[:,:-1]],1)
                Y_hat, _ = net(X, dec_input, X_valid_len)
                l = loss(Y_hat, Y)
                l.sum().backward()  # Make the loss scalar for `backward`
                d2l.grad_clipping(net, 1)
                num_tokens = Y_valid_len.sum()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l.sum(), num_tokens)
        #We evaluate the model over the validation set
        net.eval()
        current_loss = 0
        for batch in val_iter:
            with torch.cuda.amp.autocast():
                X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
                dec_input = torch.cat([X[:,-1,0].reshape(-1, 1), Y[:,:-1]],1)
                Y_hat, _ = net(X, dec_input, X_valid_len)
                with torch.no_grad():
                    current_loss += loss(Y_hat, Y).sum()
            if (current_loss < best_model_loss):
                torch.save(net.state_dict(), os.path.join("./results/models/", name + '_best-model-parameters.pt'))
            if (epoch + 1) % 10 == 0:
                losses.append((epoch+1, (metric[0] / metric[1],)))
                #animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
    plt.plot(*zip(*losses))
    plt.savefig(execution_name+'.png')

    return losses