import typing as t
from torch import nn
 

def make_sequential_model(input_size: int, layer_widths: int) -> t.List[nn.Module]:
  """Creates a sequential Pytorch model with ReLU activations, except the output layer which has no activation function 
  
  Args:
      input_size (int): Size of model input
      layer_widths (int): List of layer widths
  
  Returns:
      t.List[nn.Module]: Sequential Pytorch model
  """
  n = len(layer_widths)
  input_sizes = [input_size] + layer_widths[0:n-1]
  output_sizes = layer_widths

  layers = []

  for input_size, output_size in zip(input_sizes, output_sizes):
    layers.append(nn.Linear(input_size, output_size))
    layers.append(nn.ReLU())

  # discard last RELU activation
  layers.pop(-1)
  return nn.Sequential(*layers)