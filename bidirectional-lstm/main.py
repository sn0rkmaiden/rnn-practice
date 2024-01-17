import numpy as np
import unidecode
import torch
import torch.nn as nn
import random
from model import BiLSTM

def sample_chars_bilstm(hidden_state_f, hidden_state_b, cell_state_f, cell_state_b, seed_idx, num_seq):
      """Sample a sequence of characters from the current model, this is primarily used for test time"""
      x = np.zeros((vocab_size, 1))
      x[seed_idx] = 1
      indices = []
      for _ in range(num_seq):
          hidden_state_f, cell_state_f, y, forget_f, input, output_f, c_hat_f, \
    hidden_state_b, cell_state_b, forget_b, input_b, output_b, c_hat_b = model.forward(x, x[::-1], hidden_state_f, hidden_state_b, cell_state_f, cell_state_b)
          prob = model.calculate_probs(y)
          idx = np.random.choice(range(vocab_size), p=prob.ravel())  # ravel() flattens the matrix
          x = np.zeros((vocab_size, 1))
          x[idx] = 1
          indices.append(idx)

      return indices


def train(model, num_epochs, hidden_size, vocab_size, seq_len):
	n, p = 0, 0
	epochs = 6_000
	learning_rate = 0.1
	smooth_loss = -np.log(1.0 / vocab_size) * seq_len # loss at iteration 0
	MAX_DATA = 1000000

	losses = []

	# memory variables for Adagrad
	mWf_f, mWi_f, mWo_f, mWc_f, mWy = np.zeros_like(model.Wf_f), np.zeros_like(model.Wi_f), np.zeros_like(model.Wo_f), np.zeros_like(model.Wcc_f), np.zeros_like(model.Wy)
	mbf_f, mbi_f, mbo_f, mbc_f, mby = np.zeros_like(model.bf_f), np.zeros_like(model.bi_f), np.zeros_like(model.bo_f), np.zeros_like(model.bcc_f), np.zeros_like(model.by)

	# backward
	mWf_b, mWi_b, mWo_b, mWc_b = np.zeros_like(model.Wf_b), np.zeros_like(model.Wi_b), np.zeros_like(model.Wo_b), np.zeros_like(model.Wcc_b)
	mbf_b, mbi_b, mbo_b, mbc_b = np.zeros_like(model.bf_b), np.zeros_like(model.bi_b), np.zeros_like(model.bo_b), np.zeros_like(model.bcc_b)

	# while p < MAX_DATA:
	for i in range(num_epochs):

	  if p + seq_len + 1 >= len(data) or n == 0:
	    hprev_f = np.zeros((hidden_size, 1)) # reset RNN memory
	    hprev_b = np.zeros((hidden_size, 1))
	    cprev_f = np.zeros((hidden_size, 1))
	    cprev_b = np.zeros((hidden_size, 1))
	    p = 0 # go from start of data

	  inputs = [char_to_ix[ch] for ch in data[p:p + seq_len]]
	  targets = [char_to_ix[ch] for ch in data[p + 1 : p + seq_len + 1]]

	  if n % 1000 == 0:
	    sample_ix = sample_chars_bilstm(hprev_f, hprev_b, cprev_f, cprev_b, inputs[0], 200)
	    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
	    print('----\n %s \n----' % (txt,))

	  loss, hprev_f, cprev_f, dWf_f, dbf_f, dWi_f, dbi_f, dWcc_f, dbcc_f, dWo_f, dbo_f, dWy, dby, \
	  hprev_b, cprev_b, dWf_b, dbf_b, dWi_b, dbi_b, dWcc_b, dbcc_b, dWo_b, dbo_b = model.backward(inputs, targets, hprev_f, hprev_b, cprev_f, cprev_b)

	  smooth_loss = smooth_loss * 0.999 + loss * 0.001
	  if n % 500 == 0:
	    print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
	    losses.append(smooth_loss)

	  for param, dparam, mem in zip([model.Wf_f, model.Wi_f, model.Wo_f, model.Wcc_f, model.Wy, model.bf_f, model.bi_f, model.bo_f, model.bcc_f, model.by],
	                                [dWf_f, dWi_f, dWo_f, dWcc_f, dWy, dbf_f, dbi_f, dbo_f, dbcc_f, dby],
	                                [mWf_f, mWi_f, mWo_f, mWc_f, mWy, mbf_f, mbi_f, mbo_f, mbc_f, mby]):
	    mem += dparam * dparam
	    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

	  # backward
	  for param, dparam, mem in zip([model.Wf_b, model.Wi_b, model.Wo_b, model.Wcc_b, model.bf_b, model.bi_b, model.bo_b, model.bcc_b],
	                                [dWf_b, dWi_b, dWo_b, dWcc_b, dbf_b, dbi_b, dbo_b, dbcc_b],
	                                [mWf_b, mWi_b, mWo_b, mWc_b, mbf_b, mbi_b, mbo_b, mbc_b]):
	    mem += dparam * dparam
	    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

	  p += seq_len
	  n += 1


def generate(letter):
	hprev_f = np.zeros((hidden_size, 1)) # reset RNN memory
	hprev_b = np.zeros((hidden_size, 1))
	cprev_f = np.zeros((hidden_size, 1))
	cprev_b = np.zeros((hidden_size, 1))
	sampled_indices = sample_chars_bilstm(hprev_f, hprev_b, cprev_f, cprev_b, char_to_ix[letter], 200)
	predicted_text = ''.join(ix_to_char[idx] for idx in sampled_indices)
	print("-------------\n%s\n-------------" % predicted_text)


filename = 'input.txt'

data = unidecode.unidecode(open(filename).read())
vocab_size = len(data)

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

hidden_size = 100
seq_len = 16
num_epochs = 1500

model = BiLSTM(hidden_size, vocab_size)

train(model, num_epochs, hidden_size, vocab_size, seq_len)

generate('C')