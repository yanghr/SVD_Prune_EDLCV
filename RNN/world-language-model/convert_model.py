# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import numpy as np

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model')
parser.add_argument('--checkpoint', type=str, default='model',
                    help='path to load pretrained model')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model1 = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
model2 = model.SVD_RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, None, args.tied).to(device)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(model, data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)



if args.pretrained:
    model1.load_state_dict(torch.load(args.checkpoint+'.pth'))
    print('Loaded model evaluation')
test_loss = evaluate(model1,test_data)
print('=' * 89)
print('| Original model | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)

model1.eval()
model2.eval()
# Model decomposition
print('Model decomposition')
# Encoder
model2.encoder.weight = model1.encoder.weight
# Layer 1
W1 = model1.rnn.weight_ih_l0.data.cpu().numpy()
W2 = model1.rnn.weight_hh_l0.data.cpu().numpy()
W = np.concatenate((W1.T, W2.T), axis=0)
P, D, Q = np.linalg.svd(W, full_matrices=False)
model2.rnn.cells[0].linear.U = torch.nn.Parameter(torch.from_numpy(P).float().to(device))
model2.rnn.cells[0].linear.Singular = torch.nn.Parameter(torch.from_numpy(D).float().to(device))
model2.rnn.cells[0].linear.V = torch.nn.Parameter(torch.from_numpy(Q).float().to(device))
B1 = model1.rnn.bias_ih_l0.data
B2 = model1.rnn.bias_hh_l0.data
B = B1+B2
model2.rnn.cells[0].linear.bias = torch.nn.Parameter(B)
# Layer 2
W1 = model1.rnn.weight_ih_l1.data.cpu().numpy()
W2 = model1.rnn.weight_hh_l1.data.cpu().numpy()
W = np.concatenate((W1.T, W2.T), axis=0)
P, D, Q = np.linalg.svd(W, full_matrices=False)
model2.rnn.cells[1].linear.U = torch.nn.Parameter(torch.from_numpy(P).float().to(device))
model2.rnn.cells[1].linear.Singular = torch.nn.Parameter(torch.from_numpy(D).float().to(device))
model2.rnn.cells[1].linear.V = torch.nn.Parameter(torch.from_numpy(Q).float().to(device))
B1 = model1.rnn.bias_ih_l1.data
B2 = model1.rnn.bias_hh_l1.data
B = B1+B2
model2.rnn.cells[1].linear.bias = torch.nn.Parameter(B)
# decoder
W = model1.decoder.weight.data.cpu().numpy()
P, D, Q = np.linalg.svd(W.T, full_matrices=False)
model2.decoder.U = torch.nn.Parameter(torch.from_numpy(P).float().to(device))
model2.decoder.Singular = torch.nn.Parameter(torch.from_numpy(D).float().to(device))
model2.decoder.V = torch.nn.Parameter(torch.from_numpy(Q).float().to(device))
model2.decoder.bias = model1.decoder.bias

total_params = 0
for name, p in model2.named_parameters():
    if p.requires_grad:
        tensor = p.data.cpu().numpy()
        print(name+': '+str(tensor.shape))
        total_params += np.prod(tensor.shape)
print('Total #Para: '+str(total_params))
print('Pruned model evaluation')


# Run on test data.
test_loss = evaluate(model2, test_data)
print('=' * 89)
print('| After decomposition | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

torch.save(model2.state_dict(), args.save+'.pth')
