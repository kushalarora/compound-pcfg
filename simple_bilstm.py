import functools
from typing import Any, Callable, Optional
from flax import linen as nn
import jax
from jax import numpy as jnp

Array = jnp.ndarray

def sequence_mask(lengths: Array, max_length: int) -> Array:
  """Computes a boolean mask over sequence positions for each given length.

  Example:
  ```
  sequence_mask([1, 2], 3)
  [[True, False, False],
   [True, True, False]]
  ```

  Args:
    lengths: The length of each sequence. <int>[batch_size]
    max_length: The width of the boolean mask. Must be >= max(lengths).

  Returns:
    A mask with shape: <bool>[batch_size, max_length] indicating which
    positions are valid for each sequence.
  """
  return jnp.arange(max_length) < jnp.expand_dims(lengths, 1)


@jax.vmap
def flip_sequences(inputs: Array, lengths: Array) -> Array:
  """Flips a sequence of inputs along the time dimension.

  This function can be used to prepare inputs for the reverse direction of a
  bidirectional LSTM. It solves the issue that, when naively flipping multiple
  padded sequences stored in a matrix, the first elements would be padding
  values for those sequences that were padded. This function keeps the padding
  at the end, while flipping the rest of the elements.

  Example:
  ```python
  inputs = [[1, 0, 0],
            [2, 3, 0]
            [4, 5, 6]]
  lengths = [1, 2, 3]
  flip_sequences(inputs, lengths) = [[1, 0, 0],
                                     [3, 2, 0],
                                     [6, 5, 4]]
  ```

  Args:
    inputs: An array of input IDs <int>[batch_size, seq_length].
    lengths: The length of each sequence <int>[batch_size].

  Returns:
    An ndarray with the flipped inputs.
  """
  max_length = inputs.shape[0]
  return jnp.flip(jnp.roll(inputs, max_length - lengths, axis=0), axis=0)


class SimpleLSTM(nn.Module):
  """A simple unidirectional LSTM."""

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    return nn.OptimizedLSTMCell()(carry, x)

  @staticmethod
  def initialize_carry(batch_dims, hidden_size):
    # Use fixed random key since default state init fn is just zeros.
    return nn.OptimizedLSTMCell.initialize_carry(
        jax.random.PRNGKey(0), batch_dims, hidden_size)


class SimpleBiLSTM(nn.Module):
  """A simple bi-directional LSTM."""
  hidden_size: int

  def setup(self):
    self.forward_lstm = SimpleLSTM()
    self.backward_lstm = SimpleLSTM()

  def __call__(self, embedded_inputs, lengths):
    batch_size = embedded_inputs.shape[0]

    # Forward LSTM.
    initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
    _, forward_outputs = self.forward_lstm(initial_state, embedded_inputs)

    # Backward LSTM.
    reversed_inputs = flip_sequences(embedded_inputs, lengths)
    initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
    _, backward_outputs = self.backward_lstm(initial_state, reversed_inputs)
    backward_outputs = flip_sequences(backward_outputs, lengths)

    # Concatenate the forward and backward representations.
    outputs = jnp.concatenate([forward_outputs, backward_outputs], -1)
    return outputs