from flax.linen import nn
import jax.numpy as jnp
import jax

import numpy as np
from PCFG_vmap_jax import PCFG
from random import shuffle
from simple_lstm import SimpleBiLSTM

class ResidualLayer(nn.Module):
  out_dim: int = 100,
  def setup(self):
    self.lin1 = nn.Dense(out_dim)
    self.lin2 = nn.Dense(out_dim)

  def __call__(self, x):
    return nn.relu(self.lin2(nn.relu(self.lin1(x)))) + x


class RootMLP(nn.Module):
  state_dim: int = 256
  nt_states: int = 10
  def setup(self):
    self.lin1 = nn.Dense(self.state_dim)
    self.res1 = ResidualLayer(out_dim=self.state_dim)
    self.res2 = ResidualLayer(out_dim=self.state_dim)
    self.lin2 = nn.Dense(self.nt_states)

  def __call__(self, x):
    return self.lin2(self.res2(self.res1(self.lin1(x))))


class VocabMLP(nn.Module):
  state_dim: int = 256
  vocab: int = 100
  def setup(self):
    self.lin1 = nn.Dense(self.state_dim)
    self.res1 = ResidualLayer(out_dim=self.state_dim)
    self.res2 = ResidualLayer(out_dim=self.state_dim)
    self.lin2 = nn.Dense(self.vocab)

  def __call__(self, x):
    return self.lin2(self.res2(self.res1(self.lin1(x))))


class VAEEncoder(nn.Module):
  def setup(self):
    self.enc_emb = nn.Embed(self.vocab, self.w_dim)
    self.lstm = SimpleBiLSTM(self.h_dim)
    self.mean_layer = nn.Dense(self.z_dim)
    self.var_layer = nn.Dense(self.z_dim)
  
  def __call__(self, x):
    emb = self.enc_emb(x)
    outputs = self.lstm(emb)
    output = jnp.max(outputs, axis=1)
    mean = self.mean_layer(output)
    var = self.var_layer(output)
    return mean, var

class CompPCFG(nn.Module):
  vocab:int = 100,
  h_dim:int = 512, 
  w_dim:int = 512,
  z_dim:int = 64,
  state_dim:int = 256, 
  t_states:int = 10,
  nt_states:int = 10,
  
  def setup(self):
    self.all_states = self.nt_states + self.t_states

    self.pcfg = PCFG(self.nt_states, self.t_states)

    self.rule_mlp = nn.Dense(self.all_states**2)
    self.root_mlp = RootMLP(state_dim=self.state_dim, nt_states=self.nt_states)
    self.vocab_mlp = VocabMLP(state_dim=self.state_dim, vocab=self.vocab)

    if z_dim > 0:
      self.enc_rnn = nn.LSTM(w_dim, h_dim, bidirectional=True, num_layers = 1, batch_first = True)
      self.enc_params = nn.Linear(h_dim*2, z_dim*2)

  def __init__(self, 
               vocab = 100,
               h_dim = 512, 
               w_dim = 512,
               z_dim = 64,
               state_dim = 256, 
               t_states = 10,
               nt_states = 10,
               ):
    super(CompPCFG, self).__init__()
    self.state_dim = state_dim
    self.t_emb = nn.Parameter(torch.randn(t_states, state_dim))
    self.nt_emb = nn.Parameter(torch.randn(nt_states, state_dim))
    self.root_emb = nn.Parameter(torch.randn(1, state_dim))
    
    self.pcfg = PCFG(nt_states, t_states)
    self.nt_states = nt_states
    self.t_states = t_states
    self.all_states = nt_states + t_states

    self.dim = state_dim
    self.z_dim = z_dim

    self.register_parameter('t_emb', self.t_emb)
    self.register_parameter('nt_emb', self.nt_emb)
    self.register_parameter('root_emb', self.root_emb)

    self.rule_mlp = nn.Linear(state_dim+z_dim, self.all_states**2)
    self.root_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                  ResidualLayer(state_dim, state_dim),
                                  ResidualLayer(state_dim, state_dim),                         
                                  nn.Linear(state_dim, self.nt_states))
    self.vocab_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim),
                                   nn.Linear(state_dim, vocab))

    if z_dim > 0:
      self.enc_emb = nn.Embedding(vocab, w_dim)
      self.enc_rnn = nn.LSTM(w_dim, h_dim, bidirectional=True, num_layers = 1, batch_first = True)
      self.enc_params = nn.Linear(h_dim*2, z_dim*2)

      
  def enc(self, x):
    emb = self.enc_emb(x)
    h, _ = self.enc_rnn(emb)    
    params = self.enc_params(h.max(1)[0])
    mean = params[:, :self.z_dim]
    logvar = params[:, self.z_dim:]    
    return mean, logvar

  def kl(self, mean, logvar):
    result =  -0.5 * (logvar - torch.pow(mean, 2)- torch.exp(logvar) + 1)
    return result

  def __call__(self, x, argmax=False, use_mean=False):
    #x : batch x n
    n = x.size(1)
    batch_size = x.size(0)
    if self.z_dim > 0:
      mean, logvar = self.enc(x)
      kl = self.kl(mean, logvar).sum(1) 
      z = mean.new(batch_size, mean.size(1)).normal_(0, 1)
      z = (0.5*logvar).exp()*z + mean    
      if use_mean:
        z = mean
      self.z = z
    else:
      self.z = torch.zeros(batch_size, 1).cuda()

    t_emb = self.t_emb
    nt_emb = self.nt_emb
    root_emb = self.root_emb

    root_emb = root_emb.expand(batch_size, self.state_dim)
    t_emb = t_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, n, self.t_states, self.state_dim)
    nt_emb = nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, self.state_dim)

    if self.z_dim > 0:
      root_emb = torch.cat([root_emb, z], 1)
      z_expand = z.unsqueeze(1).expand(batch_size, n, self.z_dim)
      z_expand = z_expand.unsqueeze(2).expand(batch_size, n, self.t_states, self.z_dim)
      t_emb = torch.cat([t_emb, z_expand], 3)
      nt_emb = torch.cat([nt_emb, z.unsqueeze(1).expand(batch_size, self.nt_states, 
                                                         self.z_dim)], 2)
    root_scores = F.log_softmax(self.root_mlp(root_emb), 1)
    unary_scores = F.log_softmax(self.vocab_mlp(t_emb), 3)
    x_expand = x.unsqueeze(2).expand(batch_size, x.size(1), self.t_states).unsqueeze(3)
    unary = torch.gather(unary_scores, 3, x_expand).squeeze(3)
    rule_score = F.log_softmax(self.rule_mlp(nt_emb), 2) # nt x t**2
    rule_scores = rule_score.view(batch_size, self.nt_states, self.all_states, self.all_states)
    log_Z = vmap(self.pcfg._inside)(unary, rule_scores, root_scores)
    if self.z_dim == 0:
      kl = torch.zeros_like(log_Z)
    if argmax:
      with torch.no_grad():
        max_score, binary_matrix, spans, tags = vmap(self.pcfg._viterbi)(unary, rule_scores, root_scores)
        import pdb; pdb.set_trace()
        self.tags = tags
      return -log_Z, kl, binary_matrix, spans
    else:
      return -log_Z, kl
