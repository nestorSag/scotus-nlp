

from src.corpus import Opinions as op_loader
from src.parsers import Opinions as op_parser, PatternHasher

import re

corpus = op_loader.load()
min_paragraph_length = 250
min_n_paragraphs = 5
corpus, labels, hashes = op_parser.parse(corpus, min_paragraph_length, min_n_paragraphs)

# corpus, labels = op_parser.extract_text(corpus, min_paragraph_length, min_n_paragraphs)
# corpus, full_citations = PatternHasher.hash_full_citations(corpus)
# short_citation_rgx = f"{PatternHasher.citation_vol_rep_rgx}{PatternHasher.citation_suffix_rgx}"
# corpus = [re.sub(short_citation_rgx, "", p) for p in corpus]
# corpus, entities = PatternHasher.hash_entities(corpus)
# hashes = dict((v, k) for k, v in full_citations.items()) | dict((v, k) for k, v in short_citations.items()) | dict((v, k) for k, v in entities.items())


