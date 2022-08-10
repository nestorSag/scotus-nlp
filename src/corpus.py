import time
import shutil
import logging
import typing as t
import os, subprocess
from pathlib import Path
from urllib.parse import urljoin
from multiprocessing import Pool
from abc import abstractclassmethod, abstractmethod, ABC


import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_FOLDER = "data"


class classproperty(object):
  # Utility class for class properties
  def __init__(self, fget):
      self.fget = fget

  def __get__(self, owner_self, owner_cls):
      return self.fget(owner_cls)

        
class DataRetriever(ABC):
  """
  Abstract class for retrieving data from a URL.
  """
  @abstractclassmethod
  def retrieve_urls(cls) -> t.List[str]:
    """
    Retrieve list of valid URLs for scrapping
    """
    raise NotImplementedError("This is an abstract method, use one of the child classes instead.")

  @abstractclassmethod
  def fetch_text(cls, url: str) -> str:
    """
    Retrieve text data from a URL.
    """
    raise NotImplementedError("This is an abstract method, use one of the child classes instead.")

  @abstractmethod
  def wrap_up(self, folder: Path):
    """
    wrap up text retrieval process
    """
    pass

  def set_up(self, folder: Path):
    """
    set up text retrieval process
    """
    # create metadata folder
    metadata_folder = Path(folder) / "metadata"
    metadata_folder.mkdir(exist_ok=True)
    self.metadata_folder = metadata_folder
  

class ScotusOpinions(DataRetriever):

  """Downloads textudal data from the U.S. Supreme Court's ScotusOpinions database (https://www.supremecourt.gov)

  """

  legacy_url = "https://www.supremecourt.gov/ScotusOpinions/USReports.aspx"
  current_url = "https://www.supremecourt.gov/ScotusOpinions/slipopinion/{year}"
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
  def retrieve_urls(cls) -> t.List[str]:
    """Return a list of pdf urls from SCOTUS opinion files
    
    Returns:
        t.List[str]: urls
    """
    logger.info("Fetching valid URLs")
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

  
  @classmethod
  def fetch_text(cls, url: str) -> str: 
    """Fetches a text from a given url pointing to a pdf file; `pdftotext` is used to convert the pdf to text as an intermedate step.

    Args:
        url (str): url to a pdf file

    Returns:
        str: text data
    """  
    if shutil.which("pdftotext") is None:
      raise Exception("pdftotext is not installed. Please install it before trying again.")
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




class FindLawOpinions(DataRetriever):
  base_url = "https://caselaw.findlaw.com/court/us-supreme-court"
  url_by_year = "https://caselaw.findlaw.com/court/us-supreme-court/years/{year}"

  def __init__(self):
    self.metadata_dfs = []

  @classmethod
  def retrieve_yearly_urls(cls) -> t.List[str]:
    """Retrieve all available year URLs from the FindLaw's SCOTUS database (https://caselaw.findlaw.com)

    Returns:
        t.List[str]: List of URLs
    """    
    page = requests.get(cls.base_url) 
    content = BeautifulSoup(page.content, 'html.parser')
    links = [link["href"] for link in content.findAll("a", href=True)]
    year_urls = [link for link in links if "years" in link]
    return year_urls
    
  def scrape_urls(self, source: str) -> t.List[str]:
    """Scrapes all available urls from a given source (https://caselaw.findlaw.com/court/us-supreme-court/years/<year>)

    Args:
        source (str): source url

    Returns:
        t.List[str]: List of urls
    """    
    page = requests.get(source)
    content = BeautifulSoup(page.content, 'html.parser')
    tables = content.find_all("table")
    if len(tables) == 0:
      logger.exception(f"No tables were found in {source}")
      return []
    main_table = tables[-1]
    try:
      df = pd.read_html(str(main_table))[0]
    except Exception as e:
      logger.exception(f"Error reading main table from {source}")
      return []
    if len(df.columns) != 3 or any(df.columns != ['Description', 'Date', 'Docket #']):
      logger.exception(f"Unexpected table format in {source}")
      return []
      
    urls = [urljoin(self.base_url, anchor["href"]) for anchor in main_table.find_all("a")]
    if len(urls) != len(df):
      logger.exception(f"Number of retrieved URLs is smaller than main table length in {source}")
      df["url"] = None
    else:
      df["url"] = urls
    self.metadata_dfs.append(df)
    return urls

  def retrieve_urls(self) -> t.List[str]:
    """Return a list of urls from findlaw.com for SCOTUS opinions
    
    Returns:
        t.List[str]: urls
    """
    logger.info("Fetching valid URLs")
    years = self.retrieve_yearly_urls()
    urls = []
    for year in years:
      time.sleep(0.25)
      year_urls = self.scrape_urls(year)
      logger.info(f"Fetched {len(year_urls)} urls from {year}")
      urls.extend(year_urls)
    return urls

  @classmethod
  def fetch_text(cls, url: str) -> str:
    """Fetches text from a given url

    Args:
        url (str): url

    Returns:
        str: text
    """    
    page = requests.get(url)
    content = BeautifulSoup(page.content, 'html.parser')
    text = str(content.find("div", {"class": "caselawcontent"}).parent)
    return text

  def set_up(self, folder):
    super().set_up(folder)
    # load current metadata file if it exists
    metadata_path = self.metadata_folder / "case_metadata.csv"
    if metadata_path.is_file():
      self.metadata_dfs = [pd.read_csv(metadata_path)]

  def wrap_up(self, folder):
    # save metadata
    metadata_df = pd.concat(self.metadata_dfs).drop_duplicates()
    metadata_df.to_csv(self.metadata_folder / "case_metadata.csv", index=False)
  



class ScotusTranscripts(DataRetriever):

  """Downloads textudal data from the U.S. Supreme Court's transcript database (https://www.supremecourt.gov)

  """

  base = "https://www.supremecourt.gov"
  legacy_url = "https://www.supremecourt.gov/oral_arguments/archived_ScotusTranscripts/{year}"
  current_url = "https://www.supremecourt.gov/oral_arguments/argument_transcript/{year}"
  years = range(1970, 2022)

  @classmethod
  def retrieve_urls(cls) -> t.List[str]:
    """Return a list of pdf urls from SCOTUS transcripts in the given year
    
    Returns:
        t.List[str]: urls
    """
    all_urls = []
    for year in cls.years:
      try:
        legacy = year < 2000
        url = cls.legacy_url if legacy else cls.current_url
        href_signature = f"/pdfs/ScotusTranscripts/{year}/" if legacy else f"/argument_ScotusTranscripts/{year}"
        page = requests.get(url.format(year=year))
        content = BeautifulSoup(page.content, 'html.parser')
        links = [link["href"] for link in content.findAll("a", href=True)]
        hrefs = [link for link in links if href_signature in link]
        if legacy:
          base_url = cls.base
          urls = [f"{base_url}{href}" for href in hrefs]
        else:
          base_url = f"https://www.supremecourt.gov/oral_arguments/argument_ScotusTranscripts/{year}/"
          urls = [f"{base_url}{href.split('/')[-1]}" for href in hrefs]
      except Exception as e:
        logger.exception(f"Error fetching urls for year {year}; skipping")
        urls = []
      all_urls.extend(urls)
    return all_urls

  @classmethod
  def fetch_text(cls, url: str) -> str: 
    return ScotusOpinions.fetch_text(url)



class Loader:
  """Utility to load text data from SCOTUS cases
  """  

  types = {
    "www.supremecourt.gov:oral-arguments": ScotusTranscripts,
    "www.supremecourt.gov:opinions": ScotusOpinions,
    "caselaw.findlaw.com:opinons": FindLawOpinions
  }

  @classproperty
  def valid_types(cls):
    return list(cls.types.keys())

  @classmethod
  def download(
    cls, 
    source_and_type: str = "caselaw.findlaw.com:opinons") -> Path:
    """Extracts text from pdf urls
    
    Args:
        source_and_type (str, optional): See Loader.valid_types for valid options. Defaults to "caselaw.findlaw.com:opinons"

    Returns:
        Path: retrieved data'sq folder path
    """

    if source_and_type not in cls.types:
      raise ValueError(f"Valid source_and_type options are {cls.valid_types}")
    
    folder = Path(DATA_FOLDER) / source_and_type
    if not folder.is_dir():
      folder.mkdir(exist_ok=True, parents=True)

    downloaded = set([self.filename_to_url(file.name) for file in cls.get_local_files(folder)])

    retriever = cls.types[source_and_type]()
    retriever.set_up(folder)
    all_urls = set(retriever.retrieve_urls())

    urls = all_urls - downloaded
    if len(urls) > 0:
      logger.info(f"Fetching online data from {len(urls)} URLs. Estimated time: { len(urls) / 60} minutes")
      for k, url in enumerate(urls):
        try:
          document = retriever.fetch_text(url)
          file_path = folder / f"{self.url_to_filename(url)}"
          with open(file_path, "w") as doc:
            doc.write(document)
          logger.info(f"saved text from {url}")
        except Exception as e:
          logger.exception(f"Error fetching text from {url}")
        # this is to avoid overwhelming the server
        time.sleep(1)
    else:
      logger.info("No new data to download")

    retriever.wrap_up(folder)
    
    return folder

  @classmethod
  def load(
    cls, 
    source_and_type: str = "caselaw.findlaw.com:opinions") -> str:
    """Get raw text data from SCOTUS cases
    
    Args:
        source_and_type (str): one of 'www.supremecourt.gov:opinions' or 'caselaw.findlaw.com:opinions'. Defaults to the latter 
    
    Returns:
        str: text corpus
    """
    corpus = []
    path = cls.download(source_and_type)
    files = cls.get_local_files()
    for file in files:
      with open(file, "r") as doc:
        corpus.append(doc.read())
    return corpus

  @classmethod
  def get_local_files(cls, folder: Path) -> t.List[Path]:
    """Retuns list of (non-nested) file paths in a given folder

    Args:
        path (Path): folder path

    Returns:
        t.List[Path]: list of Path objects
    """    
    return [obj for obj in folder.iterdir() if obj.is_file()]

  @classmethod
  def url_to_filename(cls, url: str) -> str:
      return url.replace("/", "|") + ".txt"

  @classmethod
  def filename_to_url(cls, filename: str) -> str:
    return re.sub(r"\.txt$", "", filename.replace("|", "/"))






















