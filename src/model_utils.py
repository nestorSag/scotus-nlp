import io
import time
import re
import typing as t
import logging
import os, subprocess
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import spacy
import contextualSpellCheck

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocToSkipgram:

  """Class for sampling skipgrams from a corpus; every word in a central word's context is used in each batch
  """
  
  def __init__(
    self, 
    corpus: t.List[str], 
    vocab: t.List[str],
    window_length: int,
    positive_batch_size: int,
    negative_batch_size: int):
    """Initialises the sampler
    
    Args:
        corpus (t.List[str]): List of documents as strings
        vocab (t.List[str]): List of words as strings
        window_length (int): context window length to either side; context is `(x-window_length,x+window_length)` for a central word index x
        positive_batch_size (int): Size of positive word pairs batch
        negative_batch_size (int): Size of negative (e.g. noise) word pairs batch
    """
    self.vocab = {word: k for k, word in enumerate(vocab)}
    self.window_length = window_length
    self.negative_batch_size = negative_batch_size
    self.positive_batch_size = positive_batch_size
    # create sequences from corpus and vocabulary
    sequences = [[self.vocab[x] for x in doc.split(" ") if x in self.vocab] for doc in corpus]
    seq_lengths = [len(x) for x in sequences]
    # drop documents that are too short
    sequences = [seq for idx, seq in enumerate(sequences) if seq_lengths[idx] >= window_length]
    seq_lengths = [l for l in seq_lengths if l >= window_length]
    # flatten corpus to word index sequence and create probability mass function for central word sampling
    wordseq = np.concatenate([np.array(seq, dtype=np.int32) for seq in sequences], axis=0)
    central_word_pmf = 1.0/len(wordseq) * np.ones((len(wordseq),), dtype=np.float32)
    cum_lengths = np.cumsum(seq_lengths)
    logger.info(f"Flattened corpus with {cum_lengths[-1]} tokens")
    # avoid sampling central words for which part of the context would be in a different document when using a concatenated corpus
    invalid_indices = np.array([np.arange(x-window_length, x+window_length) for x in cum_lengths]).reshape(-1)
    invalid_indices = invalid_indices[invalid_indices < cum_lengths[-1]]
    central_word_pmf[invalid_indices] = 0
    central_word_pmf /= np.sum(central_word_pmf)
    self.central_word_pmf = central_word_pmf
    self.wordseq = wordseq
    # build negative sampling distribution
    seq_dtm = CountVectorizer(stop_words=None, vocabulary=vocab).fit_transform(corpus)
    m, n = len(corpus), len(self.vocab)
    word_counts = seq_dtm.T.dot(np.ones((m,1))).reshape(-1)
    word_counts /= np.sum(word_counts)
    neg_pmf = word_counts**(3/4)
    neg_pmf /= np.sum(neg_pmf)
    self.negative_sample_pmf = neg_pmf
    
  def __iter__(self):
    x_p, y_p = self.get_positive_samples()
    x_n, y_n = self.get_negative_samples()
    return np.concatenate([x_p, x_n], axis=0), np.concatenate([y_p, y_n], axi=0)

  def get_positive_samples(self) -> t.Tuple[np.ndarray, np.ndarray]:
    """Returns features and label arrays; the feature matrix has columns for word and context vocabulary indices
    
    Returns:
        t.Tuple[np.ndarray, np.ndarray]: skipgram matrix
    """

    l = self.window_length
    # sample central words
    n_words = int(np.ceil(self.positive_batch_size/(2*self.window_length)))
    central_words, central_words_pos = self.sample_central_words(n_words)
    context = self.sample_context(central_words_pos)
    # create skipgrams from all words in context window
    word = np.concatenate([word * np.ones((2*l,)) for word in central_words], axis=0)
    skipgrams = np.stack([word, context], axis=1)
    # remove skipgrams where context word is center word
    n_skipgrams = len(skipgrams)
    skipgrams = skipgrams[np.arange(n_skipgrams)%2*l != l,:]
    labels = np.ones((len(skipgrams),), dtype=np.int32)
    return skipgrams, labels 
  
  def get_negative_samples(self) -> t.Tuple[np.ndarray, np.ndarray]:
    """returns features and label arrays; the feature matrix has columns for word and context vocabulary indices
    
    Returns:
        nt.Tuple[np.ndarray, np.ndarray]: skipgram matrix
    """

    central_words, _ = self.sample_central_words(self.negative_batch_size)
    random_context = np.random.choice(np.arange(len(self.vocab)), size=self.negative_batch_size, p=self.negative_sample_pmf)
    skipgrams = np.stack([central_words, random_context], axis=1)
    labels = np.zeros((len(skipgrams),), dtype=np.int32)
    return skipgrams, labels
  
  def sample_central_words(self, n: int) -> t.Tuple[np.ndarray, np.ndarray]:
    """Returns vocabulary and corpus position indices of sampeld central words
    
    Args:
        n (int): Number of central words to sample
    
    Returns:
        t.Tuple[np.ndarray, np.ndarray]: vocabulary and corpus position indices
    """

    idx = np.random.choice(len(self.wordseq), size=n, replace=False, p=self.central_word_pmf)
    return self.wordseq[idx], idx
  
  def sample_context(self, positions: int) -> np.ndarray:
    """Returns vocabulary indices for sampled contexts from an iterable of central word positions
    
    Args:
        positions (int): Iterable of central word positions
    
    Returns:
        np.ndarray: vocabulary indices of context words
    """

    l = self.window_length
    n = len(self.wordseq)
    context_pos = np.concatenate([np.arange(max(0,pos-l), min(pos + l, n)) for pos in positions], axis=0)
    return self.wordseq[context_pos]