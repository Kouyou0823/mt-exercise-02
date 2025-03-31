import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2')
parser.add_argument('--model', type=str, default='LSTM')
parser.add_argument('--emsize', type=int, default=200)
parser.add_argument('--nhid', type=int, default=200)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--lr', type=float, default=20)
parser.add_argument('--clip', type=float, default=0.25)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=20, metavar='N')
parser.add_argument('--bptt', type=int, default=35)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--tied', action='store_true')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--mps', action='store_true', default=False)
parser.add_argument('--log-interval', type=int, default=200, metavar='N')
parser.add_argument('--save', type=str, default='model.pt')
parser.add_argument('--onnx-export', type=str, default='')
parser.add_argument('--nhead', type=int, default=2)
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--log-file', type=str, default='',
                    help='optional file to save epoch-wise perplexity results')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, consider using --cuda.")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not args.mps:
    print("WARNING: You have an MPS device, consider using --mps.")

use_mps = args.mps and torch.backends.mps.is_available()
device = torch.device("cuda" if args.cuda else "mps" if use_mps else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
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
if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    return h.detach() if isinstance(h, torch.Tensor) else tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data).view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data).view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)
        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break
    return total_loss / (len(train_data) // args.bptt)

def export_onnx(path, batch_size, seq_len):
    print('Exporting ONNX model to', os.path.realpath(path))
    model.eval()
    dummy_input = torch.zeros(seq_len, batch_size, dtype=torch.long).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

###############################################################################
# Main loop
###############################################################################

lr = args.lr
best_val_loss = None

try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train()
        val_loss = evaluate(val_data)
        test_loss = evaluate(test_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
              'valid loss {:5.2f} | test loss {:5.2f} | '
              'valid ppl {:8.2f} | test ppl {:8.2f}'.format(
            epoch, (time.time() - epoch_start_time), train_loss, val_loss, test_loss,
            math.exp(val_loss), math.exp(test_loss)))
        print('-' * 89)

        if args.log_file:
            with open(args.log_file, 'a') as f:
                f.write(f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{test_loss:.4f}\t{math.exp(val_loss):.2f}\t{math.exp(test_loss):.2f}\n")

        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            lr /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

with open(args.save, 'rb') as f:
    model = torch.load(f)
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
