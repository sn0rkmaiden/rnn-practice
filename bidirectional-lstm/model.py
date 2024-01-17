import torch
import torch.nn as nn
import numpy as np

class BiLSTM(nn.Module):
  def __init__(self, hidden_size, vocab_size):
    super(BiLSTM, self).__init__()

    self.vocab_size = vocab_size
    self.hidden_size = hidden_size

    # forward
    self.Wf_f = np.random.randn(self.hidden_size, self.hidden_size + self.vocab_size) * 0.01
    self.bf_f = np.zeros((self.hidden_size, 1))
    self.Wi_f = np.random.randn(self.hidden_size, self.hidden_size + self.vocab_size) * 0.01
    self.bi_f = np.zeros((self.hidden_size, 1))
    self.Wcc_f = np.random.randn(self.hidden_size, self.hidden_size + self.vocab_size) * 0.01
    self.bcc_f = np.zeros((self.hidden_size, 1))
    self.Wo_f = np.random.randn(self.hidden_size, self.hidden_size + self.vocab_size) * 0.01
    self.bo_f = np.zeros((self.hidden_size, 1))
    self.Wy = np.random.randn(self.vocab_size, 2 * self.hidden_size) * 0.01
    self.by = np.zeros((self.vocab_size, 1))

    # backward
    self.Wf_b = np.random.randn(self.hidden_size, self.hidden_size + self.vocab_size) * 0.01
    self.bf_b = np.zeros((self.hidden_size, 1))
    self.Wi_b = np.random.randn(self.hidden_size, self.hidden_size + self.vocab_size) * 0.01
    self.bi_b = np.zeros((self.hidden_size, 1))
    self.Wcc_b = np.random.randn(self.hidden_size, self.hidden_size + self.vocab_size) * 0.01
    self.bcc_b = np.zeros((self.hidden_size, 1))
    self.Wo_b = np.random.randn(self.hidden_size, self.hidden_size + self.vocab_size) * 0.01
    self.bo_b = np.zeros((self.hidden_size, 1))

  def sigmoid(self, z):
    with np.errstate(over='ignore', invalid='ignore'):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

  def forward(self, x_f, x_b, hprev_f, hprev_b, cprev_f, cprev_b):

    # forward
    xh_f = np.vstack((x_f, hprev_f))

    forget_f = self.sigmoid(np.dot(self.Wf_f, xh_f) + self.bf_f)
    input_f = self.sigmoid(np.dot(self.Wi_f, xh_f) + self.bi_f)
    output_f = self.sigmoid(np.dot(self.Wo_f, xh_f) + self.bo_f)
    c_hat_f = np.tanh(np.dot(self.Wcc_f, xh_f) + self.bcc_f)

    next_c_f = forget_f * cprev_f + input_f * c_hat_f
    next_h_f = output_f * np.tanh(next_c_f)
    # y = np.dot(self.Wy, next_h_f) + self.by

    # backward
    xh_b = np.vstack((x_b, hprev_b))

    forget_b = self.sigmoid(np.dot(self.Wf_b, xh_b) + self.bf_b)
    input_b = self.sigmoid(np.dot(self.Wi_b, xh_b) + self.bi_b)
    output_b = self.sigmoid(np.dot(self.Wo_b, xh_b) + self.bo_b)
    c_hat_b = np.tanh(np.dot(self.Wcc_b, xh_b) + self.bcc_b)

    next_c_b = forget_b * cprev_b + input_b * c_hat_b
    next_h_b = output_b * np.tanh(next_c_f)

    next_h = np.concatenate((next_h_f, next_h_b))
    # next_c = np.concatenate((next_c_f, next_c_b))
    y = np.dot(self.Wy, next_h) + self.by

    return next_h_f, next_c_f, y, forget_f, input_f, output_f, c_hat_f, \
    next_h_b, next_c_b, forget_b, input_b, output_b, c_hat_b

  def calculate_probs(self, y):
    return np.exp(y) / np.sum(np.exp(y))

  def backward(self, inputs, targets, hprev_f=None, hprev_b=None, cprev_f=None, cprev_b=None):
    loss = 0
    # forward
    xs, xhs_f, ys_f, hs_f, cs_f, fgs_f, igs_f, ccs_f, ogs_f = (
            {}, {}, {}, {}, {}, {}, {}, {}, {})

    # backward
    xhs_b, hs_b, cs_b, fgs_b, igs_b, ccs_b, ogs_b = (
            {}, {}, {}, {}, {}, {}, {})
    
    hs, ps, ys = ({}, {}, {})

    if hprev_f is not None:
      hs_f[-1] = np.copy(hprev_f)

    if hprev_b is not None:
      hs_b[-1] = np.copy(hprev_b)

    if cprev_f is not None:
      cs_f[-1] = np.copy(cprev_f)

    if cprev_b is not None:
      cs_b[-1] = np.copy(cprev_b)

    for t in range(len(inputs)):

      xs[t] = np.zeros((self.vocab_size, 1))
      xs[t][inputs[t]] = 1

      xs[len(inputs)-t-1] = np.zeros((self.vocab_size, 1))
      xs[len(inputs)-t-1][inputs[t]] = 1

      xhs_f[t] = np.vstack((xs[t], hs_f[t-1]))
      xhs_b[t] = np.vstack((xs[t], hs_b[t-1]))

      hs_f[t], cs_f[t], ys[t], fgs_f[t], igs_f[t], ogs_f[t], ccs_f[t], hs_b[t], cs_b[t], fgs_b[t], igs_b[t], ogs_b[t], ccs_b[t] = \
      self.forward(xs[t], xs[len(inputs)-t-1], hs_f[t-1], hs_b[t-1], cs_f[t-1], cs_b[t-1])

      ps[t] = self.calculate_probs(ys[t])

      loss += -np.log(ps[t][targets[t], 0])

      hs[t] = np.vstack((hs_f[t], hs_b[t]))

    # return loss, hs[len(inputs)-1], cs[len(inputs)-1], xs, xhs, hs, cs, ps, fgs, igs, ogs, ccs

    # forward
    dWf_f = np.zeros_like(self.Wf_f)
    dbf_f = np.zeros_like(self.bf_f)
    dWi_f = np.zeros_like(self.Wi_f)
    dbi_f = np.zeros_like(self.bi_f)
    dWcc_f = np.zeros_like(self.Wcc_f)
    dbcc_f = np.zeros_like(self.bcc_f)
    dWo_f = np.zeros_like(self.Wo_f)
    dbo_f = np.zeros_like(self.bo_f)

    # backward
    dWf_b = np.zeros_like(self.Wf_b)
    dbf_b = np.zeros_like(self.bf_b)
    dWi_b = np.zeros_like(self.Wi_b)
    dbi_b = np.zeros_like(self.bi_b)
    dWcc_b = np.zeros_like(self.Wcc_b)
    dbcc_b = np.zeros_like(self.bcc_b)
    dWo_b = np.zeros_like(self.Wo_b)
    dbo_b = np.zeros_like(self.bo_b)

    dWy = np.zeros_like(self.Wy)
    dby = np.zeros_like(self.by)

    dhnext_f = np.zeros_like(hs_f[0])
    dcnext_f = np.zeros_like(cs_f[0])

    dhnext_b = np.zeros_like(hs_f[0])
    dcnext_b = np.zeros_like(cs_f[0])

    for t in reversed(range(len(inputs))):
      # Backprop through the gradients of loss and softmax.
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1

        # Compute gradients for the Wy and by parameters.
        dWy += np.dot(dy, hs[t].T)
        dby += dy

        # Backprop through the fully-connected layer (Wy, by) to h. Also add up
        # the incoming gradient for h from the next cell.
        dh_f = np.dot(self.Wy.T[:self.hidden_size], dy) + dhnext_f
        dh_b = np.dot(self.Wy.T[self.hidden_size:], dy) + dhnext_b

        # Backprop through multiplication with output gate; here "dtanh" means
        # the gradient at the output of tanh.
        dctanh_f = ogs_f[t] * dh_f
        dctanh_b = ogs_b[t] * dh_b

        # Backprop through the tanh function; since cs[t] branches in two
        # directions we add dcnext too.
        dc_f = dctanh_f * (1 - np.tanh(cs_f[t]) ** 2) + dcnext_f
        dc_b = dctanh_b * (1 - np.tanh(cs_b[t]) ** 2) + dcnext_b

        # Backprop through multiplication with the tanh; here "dhogs" means
        # the gradient at the output of the sigmoid of the output gate. Then
        # backprop through the sigmoid itself (ogs[t] is the sigmoid output).
        dhogs_f = dh_f * np.tanh(cs_f[t])
        dho_f = dhogs_f * ogs_f[t] * (1 - ogs_f[t])

        dhogs_b = dh_b * np.tanh(cs_b[t])
        dho_b = dhogs_b * ogs_b[t] * (1 - ogs_b[t])

        # Compute gradients for the output gate parameters.
        dWo_f += np.dot(dho_f, xhs_f[t].T)
        dbo_f += dho_f

        dWo_b += np.dot(dho_b, xhs_b[t].T)
        dbo_b += dho_b

        # Backprop dho to the xh input.
        dxh_from_o_f = np.dot(self.Wo_f.T, dho_f)
        dxh_from_o_b = np.dot(self.Wo_b.T, dho_b)

        # Backprop through the forget gate: sigmoid and elementwise mul.
        dhf_f = cs_f[t-1] * dc_f * fgs_f[t] * (1 - fgs_f[t])
        dWf_f += np.dot(dhf_f, xhs_f[t].T)
        dbf_f += dhf_f
        dxh_from_f_f = np.dot(self.Wf_f.T, dhf_f)

        dhf_b = cs_b[t-1] * dc_b * fgs_b[t] * (1 - fgs_b[t])
        dWf_b += np.dot(dhf_b, xhs_b[t].T)
        dbf_b += dhf_b
        dxh_from_f_b = np.dot(self.Wf_b.T, dhf_b)

        # Backprop through the input gate: sigmoid and elementwise mul.
        dhi_f = ccs_f[t] * dc_f * igs_f[t] * (1 - igs_f[t])
        dWi_f += np.dot(dhi_f, xhs_f[t].T)
        dbi_f += dhi_f
        dxh_from_i_f = np.dot(self.Wi_f.T, dhi_f)

        dhi_b = ccs_b[t] * dc_b * igs_b[t] * (1 - igs_b[t])
        dWi_b += np.dot(dhi_b, xhs_b[t].T)
        dbi_b += dhi_b
        dxh_from_i_b = np.dot(self.Wi_b.T, dhi_b)

        dhcc_f = igs_f[t] * dc_f * (1 - ccs_f[t] ** 2)
        dWcc_f += np.dot(dhcc_f, xhs_f[t].T)
        dbcc_f += dhcc_f
        dxh_from_cc_f = np.dot(self.Wcc_f.T, dhcc_f)

        dhcc_b = igs_b[t] * dc_b * (1 - ccs_b[t] ** 2)
        dWcc_b += np.dot(dhcc_b, xhs_b[t].T)
        dbcc_b += dhcc_b
        dxh_from_cc_b= np.dot(self.Wcc_b.T, dhcc_b)

        # Combine all contributions to dxh, and extract the gradient for the
        # h part to propagate backwards as dhnext.
        dxh_f = dxh_from_o_f + dxh_from_f_f + dxh_from_i_f + dxh_from_cc_f
        dhnext_f = dxh_f[self.vocab_size:, :]

        dxh_b = dxh_from_o_b + dxh_from_f_b + dxh_from_i_b + dxh_from_cc_b
        dhnext_b = dxh_b[self.vocab_size:, :]

        # dcnext from dc and the forget gate.
        dcnext_f = fgs_f[t] * dc_f
        dcnext_b = fgs_b[t] * dc_b

    # Gradient clipping to the range [-5, 5].
    for dparam in [dWf_f, dbf_f, dWi_f, dbi_f, dWcc_f, dbcc_f, dWo_f, dbo_f, dWy, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    # backward
    for dparam in [dWf_b, dbf_b, dWi_b, dbi_b, dWcc_b, dbcc_b, dWo_b, dbo_b]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, hs_f[len(inputs)-1], cs_f[len(inputs)-1], dWf_f, dbf_f, dWi_f, dbi_f, dWcc_f, dbcc_f, dWo_f, dbo_f, dWy, dby, \
    hs_b[len(inputs)-1], cs_b[len(inputs)-1], dWf_b, dbf_b, dWi_b, dbi_b, dWcc_b, dbcc_b, dWo_b, dbo_b