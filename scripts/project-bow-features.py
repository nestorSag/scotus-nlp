import logging
import typing as t
from pathlib import Path
from multiprocessing import Pool

from src.corpus import get_clean_transcripts

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

from src.models.autoencoders import AngularAutoencoder
from src.models.utils import make_sequential_model, make_bn_sequential_model

from torch.utils.data import DataLoader
import torch.optim as optim
import torch

import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

clean_texts = get_clean_transcripts(n_cores=5)

vectorizer = CountVectorizer(
  min_df = 0.05, 
  max_df = 0.95)

# use standardised document-term matrix as feature matrix
dtm = np.asarray(vectorizer
  .fit_transform(clean_texts)
  .todense())

features = (StandardScaler()
  .fit_transform(dtm)
  .astype(np.float32))

# Create autoencoder architecture
_, input_size = features.shape
bottleneck_size = 1
polar_input_size = input_size - 1

encoder_layer_widths = [2500, 1600, 500, 200, bottleneck_size]
# encoder layers are mirrored in decoder
decoder_layer_widths = encoder_layer_widths[::-1][1::] + [polar_input_size]

model = AngularAutoencoder(
  encoder = make_bn_sequential_model(polar_input_size, encoder_layer_widths),
  decoder = make_bn_sequential_model(bottleneck_size, decoder_layer_widths),
  encoder_output_length = bottleneck_size,
  ciclicity_weight=0.1)

batch_sampler = DataLoader(
  dataset = features,
  batch_size = 64,
  shuffle=True)

opt = optim.Adam(model.parameters())

model.fit(
  data=batch_sampler,
  epochs = 150,
  opt=opt)


angles = model.project(features).reshape(-1)
radius = np.sqrt(np.diag(features.dot(features.T)))
radius /= np.std(radius)

x = (radius * np.cos(angles)).reshape(-1)
y = (radius * np.sin(angles)).reshape(-1)

plt.scatter(x, y)
plt.show()