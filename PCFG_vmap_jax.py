#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import random
import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp
from jax.tree_util import Partial

class PCFG(nn.Module):
  def __init__(self, nt_states, t_states):
    super(PCFG, self).__init__()
    self.nt_states = nt_states # non-terminal states (NT)
    self.t_states = t_states # terminal states (T)
    self.states = nt_states + t_states # total_states(NT+T) = terminal states (T) + non-terminal states (NT)

    self.huge = 1e9

  def logadd(self, x, y):
    d = torch.max(x,y)  
    return torch.log(torch.exp(x-d) + torch.exp(y-d)) + d    

  def logsumexp(self, x, dim=1):
    d = torch.max(x, dim)[0]    
    if x.dim() == 1:
      return torch.log(torch.exp(x - d).sum(dim)) + d
    else:
      return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim)) + d


  def _inside_jax(self, unary_scores, rules_scores, root_scores, do_vmap=False):
    n = unary_scores.size(0)
    alpha = unary_scores.new_zeros(n, n, self.states).fill_(-self.huge)

    for k in range(n):
      for state in range(self.t_states):
        alpha[k, k, self.nt_states + state] = unary_scores[k, state]

    def _inside_base(i, j, A, k, B, C):
      r_A_B_C = rules_scores[A, B, C]

      alpha_left = alpha[i, k, B]
      alpha_right = alpha[k+1, j, C]

      log_i_j_k_A_B_C = r_A_B_C + alpha_left + alpha_right

      return log_i_j_k_A_B_C

    for l in np.arange(1, n+1):
      if not do_vmap:
        def inside_over_length(l):
          for i in range(n):
            j = i + l

          
          tmp_u = []
          for k in np.arange(i, j):
            pass

      else:
        # in_axes_over_k = (i=None, j=None, A=None, k=None, B=None, C=0)
        in_axes_over_C = (None, None, None, None, None, 0)
        _inside_vmap_over_C = vmap(_inside_base, in_axes=in_axes_over_C)

        # in_axes_over_B = (i=None, j=None, A=None, k=None, B=0, C=None)
        in_axes_over_B = (None, None, None, None, 0, None)
        _inside_vmap_over_B_C = vmap(_inside_vmap_over_C, 
                                      in_axes=in_axes_over_B)

        # in_axes_over_k = (i=None, j=None, A=None, B=None, C=None, k=0)
        in_axes_over_k = (None, None, None, None, None, 0)

        def loop_over_k(i, j, A, k):
          if i == k:
            # If the span is of length 1, then only pre-terminals parents exist.
            l_start = self.nt_states
            l_end = self.states
          else:
            l_start = 0
            l_end = self.nt_states
          
          if k+1 == j:
            # If the span is of length 1, then only pre-terminals parents exist.
            r_start = self.nt_states
            r_end = self.states
          else: 
            r_start = 0
            r_end = self.nt_states

          return _inside_vmap_over_B_C(i, j, A, k, 
                                np.arange(l_start, l_end),
                                np.arange(r_start, r_end),)

        _inside_vmap_over_B_C_k = vmap(loop_over_k, (None, None, None, 0))

        def compute_alpha_i_j_A(i, j, A):
            if j > n - 1:
              raise AssertionError(f"j: {j} <= {n-1}")

            # shape
            log_i_j_As = _inside_vmap_over_B_C_k(i, j, A, 
                                                  np.arange(i, j))
            
            return logsumexp(log_i_j_As, axis=(3, 4, 5))

        alpha_i_j_A = compute_alpha_i_j_A(i, j, A)

        # do index update. 


    return None
  def _inside(self, unary_scores, rule_scores, root_scores):
    #inside step
    #unary scores : n x T 
    #rule scores : NT  x (NT+T) x (NT+T) ()
    #root : NT

    # sequence length (n)
    n = unary_scores.size(0)

    alpha = unary_scores.new_zeros(n, n, self.states).fill_(-self.huge)

    for k in range(n):
      for state in range(self.t_states):
        alpha[k, k, self.nt_states + state] = unary_scores[k, state]

    for i in range(n):
      for l in np.arange(1, n+1):
        j = i + l
        if j > n-1:
          break

        tmp_u = []
        for k in np.arange(i, j):
          if i == k:
            # If the span is of length 1, then only pre-terminals parents exist.
            l_start = self.nt_states
            l_end = self.states
          else:
            l_start = 0
            l_end = self.nt_states
          
          if k+1 == j:
            # If the span is of length 1, then only pre-terminals parents exist.
            r_start = self.nt_states
            r_end = self.states
          else: 
            r_start = 0
            r_end = self.nt_states
          
          # P(A..., B...->C...)
          tmp_rule_scores = rule_scores[:, l_start:l_end, r_start:r_end] # NT x NT+T X NT+T

          # log(\alpha(i, k, B...))
          alpha_left = alpha[i, k, l_start:l_end] #  (NT + T)
          alpha_left = alpha_left.unsqueeze(1).unsqueeze(0) # 1 x (NT+T) x 1

          # log(\alpha(k+1, j, C...))
          alpha_right = alpha[k+1, j, r_start:r_end] # NT
          alpha_right = alpha_right.unsqueeze(0).unsqueeze(1) # 1 x 1 x (NT+T)

          # log(\alpha(i, k, j, A... -> B..., C...)) = \
          # log(\alpha(i, k, B...)) + log(\alpha(k+1, j, C...)) + log(p(A... -> B..., C...))
          tmp_scores = alpha_left + alpha_right + tmp_rule_scores # NT x NT+T x NT+T
          tmp_scores = tmp_scores.view(self.nt_states, -1)

          tmp_u.append(self.logsumexp(tmp_scores, 1).unsqueeze(1))

        tmp_u = torch.cat(tmp_u, 1)
        tmp_u = self.logsumexp(tmp_u, 1)
        
        # TODO: Replace by index_update
        # log(\alpha(i, j, A...))
        alpha[i, j, :self.nt_states] = tmp_u[:self.nt_states]

    # log(\alpha(0, n-1, A...))
    log_Z = alpha[0, n-1, :self.nt_states] + root_scores

    # \alpha(0, n-1)
    log_Z = self.logsumexp(log_Z, 0)
    return log_Z

  def _viterbi(self, unary_scores, rule_scores, root_scores):
    #unary scores :n x T
    #rule scores :NT x (NT+T) x (NT+T)

    # sequence length (n)
    n = unary_scores.size(0)

    scores = unary_scores.new_zeros(n, n, self.states).fill_(-self.huge)
    bp = unary_scores.new_zeros(n, n, self.states).fill_(-1)
    left_bp = unary_scores.new_zeros(n, n, self.states).fill_(-1)
    right_bp = unary_scores.new_zeros(n, n, self.states).fill_(-1)
    argmax = unary_scores.new_zeros(n, n).fill_(-1)
    argmax_tags = unary_scores.new_zeros(n).fill_(-1)
    spans = []
    for k in range(n):
      for state in range(self.t_states):
        scores[k, k, self.nt_states + state] = unary_scores[k, state]        

    for i in range(n):
      for l in np.arange(1, n+1):
        j = i + l
        if j > n-1:
          break

        tmp_max_score = []
        tmp_left_child = []
        tmp_right_child = []

        for k in np.arange(i, j):
          if i == k:
            l_start = self.nt_states
            l_end = self.states
          else:
            l_start = 0
            l_end = self.nt_states

          if k+1 == j:
            r_start = self.nt_states
            r_end = self.states
          else: 
            r_start = 0
            r_end = self.nt_states

          tmp_rule_scores = rule_scores[:, l_start:l_end, r_start:r_end] # NT x NT+T X NT+T

          beta_left = scores[i, k, l_start:l_end] #  NT
          beta_left = beta_left.unsqueeze(1).unsqueeze(0) # 1 x (NT+T) x 1

          beta_right = scores[k+1, j, r_start:r_end] #  NT
          beta_right = beta_right.unsqueeze(0).unsqueeze(1) # 1 x 1 x (NT+T) 


          tmp_scores = beta_left + beta_right + tmp_rule_scores # NT x NT+T x NT+T

          tmp_scores_flat = tmp_scores.view(self.nt_states, -1)
          max_score, max_idx = torch.max(tmp_scores_flat, -1) # NT
          tmp_max_score.append(max_score.unsqueeze(1)) # NT x 1
          
          # Using rstates as it can be 60 (NT) or 90 (T+NT).
          r_states = tmp_scores.size(2)
          
          left_child = (max_idx.float() / r_states).floor().long()
          tmp_left_child.append(left_child.unsqueeze(1) + l_start) # NT x 1

          right_child = torch.remainder(max_idx, r_states)
          tmp_right_child.append(right_child.unsqueeze(1) + r_start) # NT x 1

        tmp_max_score = torch.cat(tmp_max_score, 1) # NT x l
        
        tmp_left_child = torch.cat(tmp_left_child, 1) # NT x l
        tmp_right_child = torch.cat(tmp_right_child, 1) # NT x l

        max_score, max_idx = torch.max(tmp_max_score, 1) # NT, NT

        max_left_child = torch.gather(tmp_left_child, 1, max_idx.unsqueeze(1)).squeeze(1) # (1,)

        max_right_child = torch.gather(tmp_right_child, 1, max_idx.unsqueeze(1)).squeeze(1) # (1,)

        scores[i, j, :self.nt_states] = max_score[:self.nt_states]
        bp[i, j, :self.nt_states] = max_idx[:self.nt_states] + i
        left_bp[i, j, :self.nt_states] = max_left_child[:self.nt_states]
        right_bp[i, j, :self.nt_states] = max_right_child[:self.nt_states]

    max_score = scores[0, n-1, :self.nt_states] + root_scores
    max_score, max_idx = torch.max(max_score, -1)

    def _backtrack(i, j, state):
      assert(i <= j)
      left_state = int(left_bp[i][j][state])
      right_state = int(right_bp[i][j][state])
      argmax[i][j] = 1
      if i == j:
        argmax_tags[i] = state - self.nt_states
        return None      
      else:
        k = int(bp[i][j][state])
        spans.insert(0, (i,j, state))
        _backtrack(i, k, left_state)
        _backtrack(k+1, j, right_state)
      return None  

    _backtrack(0, n-1, max_idx)
    return scores[0, n-1, 0], argmax, spans, argmax_tags


