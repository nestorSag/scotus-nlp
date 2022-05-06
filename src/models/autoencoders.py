import typing as t
import logging

import numpy as np 

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Autoencoder:
  """Autoencoder class. It is instantiated by passing custom encoder and decoder models
  
  
  Attributes:
      encoder (nn.Module): Encoder model
      decoder (nn.Module): Decoder model
      loss_trace (list): list of loss values
  """


  def __init__(
    self,
    encoder: nn.Module, 
    decoder: nn.Module):
    """Instantiates a model
    
    Args:
        encoder (nn.Module): Encoder model
        decoder (nn.Module): Decoder model
    """
    self.encoder = encoder
    self.decoder = decoder
    self.loss_trace = []
    self.loss_function = nn.MSELoss()
    #
  def eval(self):
    """Sets both models in evaluation mode
    """
    self.encoder.eval()
    self.decoder.eval()

  def train(self):
    """Sets both models in train mode
    """
    self.encoder.train()
    self.decoder.train()

  def parameters(self) -> t.List[torch.Tensor]:
    """Returns concatenated parameters from both models
    
    Returns:
        TYPE: t.List[torch.Tensor]
    """
    return list(self.encoder.parameters()) + list(self.decoder.parameters())
  
  def forward(self,data: torch.Tensor) -> torch.Tensor:
    """Pass data through encoder and decoder
    
    Args:
        data (torch.Tensor): Input data
    
    Returns:
        torch.Tensor: decoded data
    """
    encoded = self.encoder.forward(batch)
    decoded = self.decoder.forward(encoded)
    return decoded

  def loss(self,batch: torch.Tensor) -> torch.Tensor:
    """Computes loss function of autoencoder model
    
    Args:
        batch (torch.Tensor): Input data
    
    Returns:
        torch.Tensor: Loss value
    """
    return self.loss_function(self.forward(batch), batch)
  #
  def fit(
    self,
    opt: torch.optim.Optimizer,
    data: DataLoader,
    epochs: int) -> None:
    
    """
    Fit autoencoder model
    
    Args:
        opt (torch.optim.Optimizer): Optimizer object
        data (DataLoader): Data loader object wrapping the dataset
        epochs (int): Number of epochs
    
    """

    self.train()
    for k in range(epochs):
      for batch_id, batch in enumerate(data):
        opt.zero_grad()
        s1 = self.loss(batch)
        s1.backward()
        opt.step()

        loss = s1.detach().numpy()
        if k%1000 == 0:
          logger.info(f"Loss: {loss}, epoch: {k}, batch: {batch_id}")
        self.loss_trace.append(loss)
  #
  def plot_loss(self):
    """
    Returns a figure of loss versus batch number
    """

    n = len(self.loss_trace)
    if n > 1:
      fig = plt.Figure(figsize=(5,5))
      plt.plot(range(n), self.loss_trace)
      plt.xlabel("Batch number")
      plt.ylabel("Loss")
      return fig








class AngularAutoencoder(Autoencoder):
  """Angular autoencoder class; it works with polar coordinates and encodes only the angular features of data. It is instantiated by passing custom encoder and decoder models.
  
  """

  def __init__(
    self,
    encoder: nn.Module,
    decoder: nn.Module,
    encoder_output_length: int,
    ciclicity_weight: float = 1.0):
    """Instantiate model
    
    Args:
        encoder (nn.Module): Encoder model
        decoder (nn.Module): Decoder model
        encoder_output_length (int): The output length of the encoder has to be passed manually
        ciclicity_weight (float, optional): Weight of the ciclicity loss. This is the part of the loss that penalises departure from equality between angles of zero and 2*pi
    
    Raises:
        ValueError: Description
    """
    self.encoder = encoder
    self.decoder = decoder
    self.encoder_output_length = encoder_output_length
    self.loss_trace = []
    if ciclicity_weight < 0:
      raise ValueError("ciclicity_weight should be non-negative")
    self.ciclicity_weight = ciclicity_weight
    self.closeness_loss = []
    self.loss_function = nn.L1Loss()

  def to_angle(self, x: torch.Tensor) -> torch.Tensor:
    """Maps input to an angle between 0 and 2*pi
    
    Args:
        x (torch.Tensor): Input data
    
    Returns:
        torch.Tensor: Angle values
    """
    angle = np.pi * torch.sigmoid(x)
    angle[:,-1] *= 2
    return angle

  def cartesian_to_polar(self, x: torch.Tensor) -> torch.Tensor:
    """Maps cartesian data to polar coordinates
    
    Args:
        x (torch.Tensor): Input data
    
    Returns:
        TYPE: Data in polar coordinates
    """
    # take data in usual Cartesian coordinates and return angular representation
    _,n = x.shape
    rev_col_idx = list(reversed(range(n)))
    cum_norms = torch.sqrt(torch.cumsum((x**2)[:,rev_col_idx],dim=-1)[:,rev_col_idx])
    std_batch = torch.acos(x/cum_norms)
    std_batch[std_batch != std_batch] = 0 #NaN are zeros by convention
    std_batch[x[:,n-1] < 0,n-2] = 2*np.pi - std_batch[x[:,n-1] < 0,n-2]

    return std_batch[:,0:n-1]

  def polar_to_cartesian(self, x: torch.Tensor) -> torch.Tensor:
    """Maps data in polar coordinates to cartesian coordinates
    
    Args:
        x (torch.Tensor): Input data
    
    Returns:
        TYPE: Data in cartesian coordinates
    """
    # since angular spaces involves long products of trigonometric functions, this method
    # handles everything in log scale to have long sums instead of long products, for higher 
    # numerical accuracy.
    
    y = x
    norms = 1

    m,n = y.shape
    X = torch.cat((np.pi/2*torch.ones(m,1),y,torch.zeros((m,1))),dim=-1) #extended angle
    cos_ = torch.cos(X)
    sin_ = torch.sin(X)
    #need to keep track of the signs and then operate everything in absolute value
    # to be able to use logarithm sums
    cos_sign = torch.sign(cos_) + (cos_==0)
    sin_sign = torch.sign(sin_) + (sin_==0)
    cos_sign[torch.abs(cos_) < 1e-7] = 1 #collapse to zero (sign 1) when value is close enough
    sin_sign[torch.abs(sin_) < 1e-7] = 1 #collapse to zero (sign 1) when value is close enough
    angle_logcos = torch.log(torch.abs(cos_))
    angle_logsin = torch.log(torch.abs(sin_))
    cum_logsin = torch.cumsum(angle_logsin,dim=-1)
    log_cartesian = cum_logsin[:,0:(n+1)] + angle_logcos[:,1::]
    #operate the signs
    cum_logsin_sign = torch.cumprod(sin_sign,dim=-1)
    cartesian_signs = cum_logsin_sign[:,0:(n+1)]*cos_sign[:,1::]
    return norms*torch.exp(log_cartesian)*cartesian_signs

  def ciclicity_loss(self, batch_size: int) -> torch.Tensor:
    """Computes the loss term that penalises the deviation from equality between decoder-mapped values of angles zero and 2*pi
    
    Returns:
        torch.Tensor: loss value
    """
    # generate a uniformly distributed random angle
    n_cols = self.encoder_output_length

    random_angles = np.pi*torch.rand((batch_size,n_cols-1))
    
    # append cyclic dimension
    cyclic_right_end = 2*np.pi*torch.ones((batch_size,1))
    cyclic_left_end = torch.zeros((batch_size,1))

    random_right_end = torch.cat((random_angles,cyclic_right_end),1)
    random_left_end = torch.cat((random_angles,cyclic_left_end),1)

    # randomly decide which end is chasing the other end
    head = np.random.uniform(size=1) > 0.5 
    if head:
      with torch.no_grad():
        r1 = self.decoder.forward(torch.Tensor(random_right_end))
      r0 =  self.decoder.forward(torch.Tensor(random_left_end))
    else:
      with torch.no_grad():
        r0 = self.decoder.forward(torch.Tensor(random_right_end))
      r1 = self.decoder.forward(torch.Tensor(random_left_end))
    ##########
    
    return self.loss_function(r0, r1)
  
  def loss(self,batch: torch.Tensor) -> torch.Tensor:

    encoded = self.encoder(batch)
    angle = self.to_angle(encoded)
    decoded = self.decoder(angle)

    reconstruction_loss =  self.loss_function(decoded, batch)

    # enforce closedness restriction in decoder in evaluation mode 
    if self.ciclicity_weight > 0:
      m, _ = batch.shape
      # self.decoder.eval()
      closeness_loss = self.ciclicity_loss(m) / m
      # self.decoder.train()
      # self.ciclicity_loss_values.append(float(closeness_loss.detach().numpy()))
      return reconstruction_loss + self.ciclicity_weight * closeness_loss
    else:
      return reconstruction_loss

  def fit(
    self,
    opt: torch.optim.Optimizer,
    data: DataLoader,
    epochs: int) -> None:
    
      """
      Fit autoencoder model
      
      Args:
        opt (torch.optim.Optimizer): Optimizer object
        data (DataLoader): Data loader object wrapping the dataset
        epochs (int): Number of epochs
      
      """

      self.train()
      for k in range(epochs):
        for batch_id, batch in enumerate(data):
          preprocessed_batch = self.cartesian_to_polar(batch)
          opt.zero_grad()
          s1 = self.loss(preprocessed_batch)
          s1.backward()
          opt.step()

          loss = s1.detach().numpy()
          if batch_id%50 == 0:
            logger.info(f"Loss: {loss}, epoch: {k}, batch: {batch_id}")
          self.loss_trace.append(loss)

  def project(self, x: np.ndarray) -> np.ndarray:
    """Project data in cartesian coordinates to learned polar representation
    
    Args:
      x (torch.Tensor): Input data
    
    Returns:
      torch.Tensor: Polar representation
    """
    if len(x.shape) == 1:
      x = x.reshape(1,-1)
    x = torch.from_numpy(x)
    self.eval()
    with torch.no_grad():
      return self.to_angle(self.encoder.forward(self.cartesian_to_polar(x))).numpy()