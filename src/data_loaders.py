from abc import abstractclassmethod, abstractmethod, ABC
from http.client import INTERNAL_SERVER_ERROR
import io
import time
import re
import gzip
import typing as t
import logging
import warnings
import traceback
import os, subprocess
from pathlib import Path
from urllib.parse import urljoin
from multiprocessing import Pool


from src.config import get_config

import requests
from bs4 import BeautifulSoup

import numpy as np

from spacy import load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# try:
#   nlp = load('en_core_web_sm')
# except:
#   print(f"Error while loading labeling model. Download the pre-trained model by running 'python -m spacy download en_core_web_sm' before trying again. Full error trace: {traceback.format_exc()}")



DATA_FOLDER = "data"
class UrlRetriever(ABC):
  """
  Abstract class for retrieving data from a URL.
  """
  @abstractclassmethod
  def retrieve(cls) -> t.List[str]:
    """
    Retrieve data from the URL.
    """
    raise NotImplementedError("This is an abstract method, use one of the child classes instead.")



class Transcripts(UrlRetriever):

  """Downloads textudal data from the U.S. Supreme Court's transcript database

  """

  base = "https://www.supremecourt.gov"
  legacy_url = "https://www.supremecourt.gov/oral_arguments/archived_transcripts/{year}"
  current_url = "https://www.supremecourt.gov/oral_arguments/argument_transcript/{year}"
  years = range(1970, 2022)

  @classmethod
  def retrieve(cls) -> t.List[str]:
    """Return a list of pdf urls from transcripts in the given year
    
    Args:
        year (int): year
    
    Returns:
        t.List[str]: urls
    """
    all_urls = []
    for year in cls.years:
      try:
        legacy = year < 2000
        url = cls.legacy_url if legacy else cls.current_url
        href_signature = f"/pdfs/transcripts/{year}/" if legacy else f"/argument_transcripts/{year}"
        page = requests.get(url.format(year=year))
        content = BeautifulSoup(page.content, 'html.parser')
        links = [link["href"] for link in content.findAll("a", href=True)]
        hrefs = [link for link in links if href_signature in link]
        if legacy:
          base_url = cls.base
          urls = [f"{base_url}{href}" for href in hrefs]
        else:
          base_url = f"https://www.supremecourt.gov/oral_arguments/argument_transcripts/{year}/"
          urls = [f"{base_url}{href.split('/')[-1]}" for href in hrefs]
      except Exception as e:
        logger.exception(f"Error fetching urls for year {year}; skipping")
        urls = []
      all_urls.extend(urls)
    return all_urls



class Opinions(UrlRetriever):

  """Downloads textudal data from the U.S. Supreme Court's opinions database

  """

  legacy_url = "https://www.supremecourt.gov/opinions/USReports.aspx"
  current_url = "https://www.supremecourt.gov/opinions/slipopinion/{year}"
  slip_opinion_years = np.arange(14,22)

  @classmethod
  def is_valid(cls, url: str) -> bool:
    """Returns a boolean of whether the link points to a valid data file

    Args:
        url (str): url

    Returns:
        bool: True if valid, False otherwise
    """    
    is_valid = (
      (
        "opinion" in url 
        or "preliminaryprint"  in url 
        or "boundvolumes" in url
      )
      and ".pdf" in url
    )

    if not is_valid and ".pdf" in url:
      logger.debug(f"url marked as invalid: {url}")
    return is_valid

  @classmethod
  def retrieve(cls) -> t.List[str]:
    """Return a list of pdf urls from transcripts in the given year
    
    Args:
        year (int): year
    
    Returns:
        t.List[str]: urls
    """
    # get legacy pdf links
    url = cls.legacy_url
    href_signature = ".pdf"
    page = requests.get(url)
    content = BeautifulSoup(page.content, 'html.parser')
    legacy_links = [link["href"] for link in content.findAll("a", href=True)]
    hrefs = [urljoin(url, link) for link in legacy_links if cls.is_valid(link)]

    # get current pdf links
    url = cls.current_url
    for year in cls.slip_opinion_years:
      page = requests.get(url.format(year=year))
      content = BeautifulSoup(page.content, 'html.parser')
      current_links = [link["href"] for link in content.findAll("a", href=True)]
      current_hrefs = [urljoin(url, link) for link in current_links if cls.is_valid(link)]
      hrefs += current_hrefs

    return hrefs


class Loader:
  """Utility to load text data from the U.S. Supreme Court's website
  """  

  types = {
    "transcripts": Transcripts, 
    "opinions": Opinions
  }

  @classmethod
  def fetch_text(cls, url: str) -> str: 
    """Fetches a text from a given url pointing to a pdf file; `pdftotext` is used to convert the pdf to text as an intermedate step.

    Args:
        url (str): url to a pdf file

    Returns:
        str: text data
    """    
    try:
      response = requests.get(url)
      filename = Path('temp.pdf')
      filename.write_bytes(response.content)
      args = ["pdftotext",
              '-enc',
              'UTF-8',
              "temp.pdf",
              '-']
      res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      text = res.stdout.decode('utf-8')
      os.remove("temp.pdf")
    except Exception as e:
      logger.exception(f"Error extracting text from {url}; skipping")
      text = None
    return text

  @classmethod
  def download(
    cls, 
    data_type: str) -> Path:
    """Extracts text from pdf urls
    
    Args:
        data_type (str): one of 'transcripts' or 'opinions'

    Returns:
        Path: data folder path
    """
    if data_type not in cls.types:
      raise ValueError(f"Valid data_type options are {cls.types.keys()}")
    
    retriever = cls.types[data_type]()
    urls = retriever.retrieve()

    folder = Path(DATA_FOLDER) / data_type
    if not folder.is_dir() or len(list(folder.iterdir())) == 0:
      folder.mkdir(exist_ok=True, parents=True)
      
      logger.info("Fetching online data. This might take a while...")
      for k, url in enumerate(urls):
        logger.info(f"Trying to fetch text from {url}")
        document = cls.fetch_text(url)
        url_name = Path(url.split("/")[-1]).stem
        file_path = folder / f"{k}_{url_name}.txt"
        logger.debug(f"Saving text in {file_path}")
        with open(file_path, "w") as doc:
          doc.write(document)
        # this is to avoid overwhelming the server
        time.sleep(1)
    else:
      logger.info(f"Data has been downloaded already; skipping")

    return folder

  @classmethod
  def load(
    cls, 
    data_type: str) -> str:
    """Get raw text data from the U.S. Supreme Court's website
    
    Args:
        data_type (str): one of 'transcripts' or 'opinions'
    
    Returns:
        str: List of text data
    """
    corpus = []
    path = cls.download(data_type)
    for file in path.iterdir():
      with open(file, "r") as doc:
        corpus.append(doc.read())
    return corpus

























# def file_to_text(file: str) -> str:
#   """Processes a transcript file into a string tuple with the raw argument transcripts and the case ID if found. Returned argument transcripts do not include preamble or afterword. ID can be None if no match is found.

#   Args:
#       file (str): Input file path

#   Returns:
#       t.TUple[str, t.Optional[str]]: document string and case ID string
#   """
#   with open(file, "r") as f:
#     lines = f.readlines()
#   # remove empty or numbering lines
#   lines = [line for line in lines if line != "" and not re.match(r"[0-9]{1,2}", line)]
#   n = len(lines)
#   # find start and end lines of the argument within document
#   start_line, end_line = (next((idx for idx in range(n) if "chief justice" in lines[idx].lower()), 0), 
#     next((idx for idx in range(n-1,-1,-1) if "the case is submitted" in lines[idx].lower()), n-1))
#   doc = " ".join(lines[start_line:end_line]) # keep oral argument section only
#   # replace linebreaks, numbers and non-ascii characters
#   doc = doc.replace("\n", " ").encode("utf-8").decode()
#   # doc = re.sub(r'[^a-z ]+','', doc)
#   # remove unnecessary spaces
#   doc = re.sub(r' +',' ', doc)
#   logger.debug(f"processed {file} ({end_line - start_line} lines)")

#   # extract case id by processing preamble
#   # collapse preamble into a single string to fix cases where id has been split

#   # id_rgx = r"No\. *(\d{2}-\d+)"
#   # preamble = " ".join(lines[0:start_line])
#   # match = re.search(id_rgx, preamble)
#   # case_id = match.group(1) if match is not None else None

#   return doc


# def get_raw_transcripts() -> t.List[str]:
#   """Get raw transcripts from oral arguments. They do not contain preamble or afterword sections.
  
#   Returns:
#       t.List[str]: List of transcript texts
#   """
#   config = get_config()
#   DATA_FOLDER = Path(config["data"]["folder"]) / "raw"
#   if not DATA_FOLDER.is_dir():
#     Transcripts.run(
#       years=range(config["data"]["scrape_from"], config["data"]["scrape_to"]+1), 
#       output_folder=DATA_FOLDER)

#   transcript_files: t.Generator[Path, None, None] = [
#   file for folder in DATA_FOLDER.iterdir() for file in folder.iterdir()
#   ]

#   logger.info(f"Processing {len(transcript_files)} files..")
#   texts = [file_to_text(file) for file in transcript_files]
#   return texts

# def clean_text(text: str):
#   """Utility function to clean and annotate strings of text
  
#   Args:
#       text (str): input text
  
#   Returns:
#       TYPE: lemmatised text without stop-words or number-like words
#   """
#   lemmatised = " ".join([token.lemma_ for token in nlp(text) if not token.is_stop and not token.like_num])
#   return lemmatised

# def get_clean_transcripts(n_cores: int) -> t.List[str]:
#   """Returns a list of lemmatised transcripts without stop-words or number-like words.
#   """
#   config = get_config()
#   DATA_FOLDER = Path(config["data"]["folder"])
#   intermediate_file = (DATA_FOLDER / "intermediate" / "lemmatised.zip")

#   if intermediate_file.is_file():
#     with gzip.open(intermediate_file, "r") as g:
#       clean_texts = g.read().decode("utf-8").split("\n")

#   else:
#     texts = get_raw_transcripts()
    
#     # use multiprocessing to speed up the process
#     logger.info("Lemmatising data..")
#     with Pool(n_cores) as p:
#       clean_texts = p.map(clean_text, texts)

#     intermediate_file.parents[0].mkdir(exist_ok=True, parents=True)
#     logger.info("Saving intermediate results for reuse..")
#     with gzip.open(intermediate_file, "wb") as g:
#       g.write("\n".join(clean_texts).encode("utf-8"))

#   warnings.warn("Order of clean texts list may not be the same as for raw texts")
#   return clean_texts