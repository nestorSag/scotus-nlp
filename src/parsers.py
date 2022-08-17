import typing as t
import logging
import hashlib
import re

from bs4 import BeautifulSoup
from src.corpus import Loader
from src.formats import ParsedOpinion

logger = logging.getLogger(__name__)

class Opinions:
    @classmethod
    def extract_text(
        cls,
        raw_corpus: str,
        min_paragraph_length: int = 250,
        min_n_paragraphs: int = 5) -> str:
        html_content = [BeautifulSoup(text, 'html.parser') for text in raw_corpus]
        cases = [x.find_all('p') for x in html_content]
        processed = [[p.text for p in case] for case in cases]
        processed = [[p for p in proc if len(p) >= min_paragraph_length] for proc in processed]
        processed = [proc for proc in processed if len(proc) >= min_n_paragraphs]
        corpus, labels = zip(*[(p, idx) for idx, doc in enumerate(processed) for p in doc])
        corpus = [re.sub(r"\n", "", p) for p in corpus]
        return corpus, labels

    @classmethod
    def parse(
        cls,
        raw_corpus: str,
        min_paragraph_length: int = 250,
        min_n_paragraphs: int = 5) -> t.Tuple[t.List[str], t.List[int], t.Dict[str,str]]:
        logger.info("Parsing HTML content")
        corpus, labels = cls.extract_text(raw_corpus, min_paragraph_length, min_n_paragraphs)
        # hash full citations
        logger.info("Hashing citations")
        corpus, full_citations = PatternHasher.hash_full_citations(corpus)
        logger.info("removing unnecessary strings")
        # remove short citations
        short_citation_rgx = f"{PatternHasher.citation_vol_rep_rgx}{PatternHasher.citation_suffix_rgx}"
        corpus = [re.sub(short_citation_rgx, "", p) for p in corpus]
        # replace numbers by token
        corpus = [re.sub(r" \d+", " NUMTOKEN", p) for p in corpus]
        # remove square brackets 
        corpus = [re.sub(r"\[|\]", "", p) for p in corpus]
        # remove content in parenthesis
        rgx = re.compile(r"\([^)]*\)")
        corpus = [re.sub(rgx, "", p) for p in corpus]
        # corpus, entities = PatternHasher.hash_entities(corpus)
        hashes = dict((v, k) for k, v in full_citations.items()) # | dict((v, k) for k, v in short_citations.items()) | dict((v, k) for k, v in entities.items())
        return corpus, labels, hashes

class PatternHasher:
    proper_name_rgx = r"(?:[A-Z][a-z'.]+(?: of the | of |, | & | \d+| and | )?)+"
    case_name_rgx = f"{proper_name_rgx} v\\. {proper_name_rgx}"
    reporters_rgx = r"U\. ?S\.|S\. ?Ct\.|I\. ?C\. ?C\.|L\. ?Ed\.|L\. ?Ed\. ?2d|F\.|F\. ?2d|F\. ?3d|F\. ?Supp\.|F\. ?Supp\. ?2d|Fed\. ?Cl\.|B\. ?R\.|T\. ?R\.|M\. ?J\.|F\. ?R\. ?D\.|Vet\. ?App\.|A\.|A\. ?2d|Cal\. ?Rptr\.|Cal\. ?Rptr\. ?2d|N\.Y\.S\.|N\. ?Y\. ?S\. ?2d|N\. ?E\.|N\. ?E\. ?2d|N\. ?W\.|N\. ?W\. ?2d|S\. ?E\.|S\. ?E\. ?2d|So\.|So\. ?2d|S\. ?W\.|S\. ?W\. ?2d|S\. ?W\. ?3d|P\.|P\. ?2d|P\. ?3d"
    citation_vol_rep_rgx = f"\\d+ (?:{reporters_rgx}) \\d+"
    citation_suffix_rgx = r"(?:, [\d-]+)*(?: \(.{1,25}\))?"
    # name_and_volume_rgx = f"{case_name_rgx}\\d+ (?:{reporters_rgx})"
    # case_citation_regex = f"{name_and_volume_rgx} \\d+(?:, [\\d-]+)?(?: \\(\\d+\\))?"
    @classmethod
    def hash_pattern(
        cls,
        corpus: t.List[str],
        match_pattern: str,
        key_pattern: t.Optional[None] = None,
        hash_preffix: str = "")-> t.Tuple[t.List[str], t.Dict[str, str]]:
        matches = [re.findall(match_pattern, p) for p in corpus]
        match2hash = {
            (re.match(key_pattern, match).group(0) if key_pattern is not None else match): 
            f"{hash_preffix}{hashlib.md5(bytes(match.encode('utf-8'))).hexdigest()}" 
            for match_list in matches for match in match_list
        }
        for k, _ in enumerate(corpus):
            for match in matches[k]:
                hashed = match2hash[re.match(key_pattern, match).group(0)]
                # replace full citation by its hash
                corpus[k] = corpus[k].replace(match, f" {hashed} ")
        return corpus, match2hash

    @classmethod
    def hash_full_citations(cls, corpus: t.List[str]) -> t.Tuple[t.List[str], t.Dict[str, str]]:
        full_citation_rgx = f"{cls.case_name_rgx}{cls.citation_vol_rep_rgx}{cls.citation_suffix_rgx}"
        case_volume_rgx = f"{cls.case_name_rgx}{cls.citation_vol_rep_rgx}"
        return cls.hash_pattern(corpus, full_citation_rgx, case_volume_rgx, "full")
    
    @classmethod
    def hash_entities(cls, corpus: t.List[str]) -> t.Tuple[t.List[str], t.Dict[str, str]]:
        return cls.hash_pattern(corpus, cls.proper_name_rgx, cls.proper_name_rgx, "short")
    
    # @classmethod
    # def hash_short_citations(cls, corpus: t.List[str]) -> t.Tuple[t.List[str], t.Dict[str, str]]:
    #     short_citation_rgx = f"{cls.citation_vol_rep_rgx}{cls.citation_suffix_rgx}"
    #     return cls.hash_pattern(corpus, short_citation_rgx, None, "ent")

# class OpinionParser:
#     """This class wraps the logic to parse  opinions and rulings from the SCOTUS corpus.

#     Returns:
#         _type_: _description_
#     """    
#     # section_delimiter = r"\nsupreme court of the united states\n"
#     docket_rgx = r"\nno\. (\d+[–-]\d+)"

#     section_delimiters = {
#         "syllabus" : (docket_rgx, r"delivered the opinion of the court|\nheld: |\nper curiam"),
#         "opinion" : (r"delivered the opinion of the court|\nheld: |\nper curiam", r"it is so ordered.\n"),
#         "concur" : (r"\njustice .+, concurring", r"\njustice .+, dissenting"),
#         "disent" : (r"\njustice .+, dissenting", r"i respectfully dissent")
#     }

#     sections = ["syllabus", "opinion", "concur", "disent"]

#     paragraph_rgx = re.compile(r".*?\.\n", flags=re.DOTALL)

#     min_lines = 30

#     @classmethod
#     def split_into_cases(cls, text: str) -> t.List[str]:
#         """Split opinion record into separate cases

#         Args:
#             text (str): opinion text data

#         Returns:
#             t.List[str]: list of case texts
#         """        
#         docket_numbers = re.findall(cls.docket_rgx, text)
#         if len(docket_numbers) > 0:
#             cases = (re
#             .compile(cls.docket_rgx)
#             .split(text)
#             )[1::] #drop preamble
#             cases = [docket_number + case for docket_number, case in zip(docket_numbers, cases)]
#             cases = [case for case in cases if len(case.split("\n")) > cls.min_lines]
#             return cases
#         else:
#             return []

#     @classmethod
#     def split_into_sections(cls, text: str) -> t.Tuple[t.List[str], str]:
#         """Splits case record into sections. See OpinionParser.sections for section types.

#         Args:
#             text (str): text string

#         Returns:
#             t.Tuple[t.List[str], str]: Tuple of section texts and docket number
#         """  
#         docket_number = re.search(cls.docket_rgx, text).group(1)
#         if docket_number is None:
#             return None
#         sections = []      
#         for section in cls.sections:
#             start_rgx, end_rgx = cls.section_delimiters[section]
#             start_match = re.search(start_rgx, text)
#             get_last_end_match = section in ["concur", "dissent"]
#             if get_last_end_match:
#                 end_match = re.findall(end_rgx, text)
#             else:
#                 end_match = re.search(end_rgx, text)
#             if start_match and end_match:
#                 start_index = start_match.start()
#                 end_index = end_match[-1].start() if get_last_end_match else end_match.start()
#                 section_text = text[start_index:end_index]
#                 sections.append(section_text)
#             else:
#                 logging.debug(f"No section '{section}' found")
#         return sections, docket_number

#     @classmethod
#     def split_into_paragraphs(cls, text: str) -> t.List[str]:
#         """Splits text into paragraphs.

#         Args:
#             text (str): text string

#         Returns:
#             t.List[str]: List of paragraphs
#         """    
#         return re.findall(cls.paragraph_rgx, text)

#     @classmethod
#     def parse(cls, texts: t.List[str]) -> t.List[ParsedOpinion]:
#         """Preprocesses a list of texts.

#         Args:
#             texts (t.List[str]): list of text strings

#         Returns:
#             t.List[OpinionsFormat]: list of preprocessed texts
#         """        
#         for text in texts:
#             text = text.lower()
#             cases = cls.split_into_cases(text)
#             for case in cases:
#                 sections, docket_number = cls.split_into_sections(case)
#                 paragraphs = [cls.split_into_paragraphs(section) for section in sections]
#                 yield ParsedOpinion(docket_number, cls.sections, paragraphs)
