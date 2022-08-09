import typing as t
import logging
import re


from src.corpus import Loader
from src.formats import ParsedOpinion

logger = logging.getLogger(__name__)

class OpinionParser:
    """This class wraps the logic to parse  opinions and rulings from the SCOTUS corpus.

    Returns:
        _type_: _description_
    """    
    # section_delimiter = r"\nsupreme court of the united states\n"
    docket_rgx = r"\nno\. (\d+[–-]\d+)"

    section_delimiters = {
        "syllabus" : (docket_rgx, r"delivered the opinion of the court|\nheld: |\nper curiam"),
        "opinion" : (r"delivered the opinion of the court|\nheld: |\nper curiam", r"it is so ordered.\n"),
        "concur" : (r"\njustice .+, concurring", r"\njustice .+, dissenting"),
        "disent" : (r"\njustice .+, dissenting", r"i respectfully dissent")
    }

    sections = ["syllabus", "opinion", "concur", "disent"]

    paragraph_rgx = re.compile(r".*?\.\n", flags=re.DOTALL)

    min_lines = 30

    @classmethod
    def split_into_cases(cls, text: str) -> t.List[str]:
        """Split opinion record into separate cases

        Args:
            text (str): opinion text data

        Returns:
            t.List[str]: list of case texts
        """        
        docket_numbers = re.findall(cls.docket_rgx, text)
        if len(docket_numbers) > 0:
            cases = (re
            .compile(cls.docket_rgx)
            .split(text)
            )[1::] #drop preamble
            cases = [docket_number + case for docket_number, case in zip(docket_numbers, cases)]
            cases = [case for case in cases if len(case.split("\n")) > cls.min_lines]
            return cases
        else:
            return []

    @classmethod
    def split_into_sections(cls, text: str) -> t.Tuple[t.List[str], str]:
        """Splits case record into sections. See OpinionParser.sections for section types.

        Args:
            text (str): text string

        Returns:
            t.Tuple[t.List[str], str]: Tuple of section texts and docket number
        """  
        docket_number = re.search(cls.docket_rgx, text).group(1)
        if docket_number is None:
            return None
        sections = []      
        for section in cls.sections:
            start_rgx, end_rgx = cls.section_delimiters[section]
            start_match = re.search(start_rgx, text)
            get_last_end_match = section in ["concur", "dissent"]
            if get_last_end_match:
                end_match = re.findall(end_rgx, text)
            else:
                end_match = re.search(end_rgx, text)
            if start_match and end_match:
                start_index = start_match.start()
                end_index = end_match[-1].start() if get_last_end_match else end_match.start()
                section_text = text[start_index:end_index]
                sections.append(section_text)
            else:
                logging.debug(f"No section '{section}' found")
        return sections, docket_number

    @classmethod
    def split_into_paragraphs(cls, text: str) -> t.List[str]:
        """Splits text into paragraphs.

        Args:
            text (str): text string

        Returns:
            t.List[str]: List of paragraphs
        """    
        return re.findall(cls.paragraph_rgx, text)

    @classmethod
    def parse(cls, texts: t.List[str]) -> t.List[ParsedOpinion]:
        """Preprocesses a list of texts.

        Args:
            texts (t.List[str]): list of text strings

        Returns:
            t.List[OpinionsFormat]: list of preprocessed texts
        """        
        for text in texts:
            text = text.lower()
            cases = cls.split_into_cases(text)
            for case in cases:
                sections, docket_number = cls.split_into_sections(case)
                paragraphs = [cls.split_into_paragraphs(section) for section in sections]
                yield ParsedOpinion(docket_number, cls.sections, paragraphs)
