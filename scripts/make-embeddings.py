import logging
import typing as t
from pathlib import Path
from multiprocessing import Pool

from src.data_loaders import get_clean_transcripts

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

import torch.optim as optim
import torch

from src.models.embeddings import Word2Vec

np.random.seed(0)
torch.manual_seed(0)

clean_texts = get_clean_transcripts(n_cores=5)

vectorizer = CountVectorizer(
  min_df = 0.05, 
  max_df = 0.95).fit(clean_texts)

vocabulary = vectorizer.vocabulary_

model = Word2Vec(
  vocabulary = vocabulary,
  embedding_size = 128)

opt = optim.Adam(model.parameters())

model.fit(
  opt = opt,
  corpus = clean_texts,
  context_size=8,
  context_batch_size=4,
  total_steps = 1000)
