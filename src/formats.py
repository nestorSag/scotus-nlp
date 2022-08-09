import typing as t

class ParsedOpinion:
    """Document wrappers that reflects the structure of SCOTUS opinions and rulings, with paragraphs subdivided into sections (see preproc.Opinion.section_order)
    """    
    def __init__(self, doc_id: str, sections: t.List[str], paragraphs: t.List[t.List[str]]):
        """
        Args:
            doc_id (str): opinion's docket number
            sections (t.List[str]): list of parsed sections
            paragraphs (t.List[t.List[str]]): list of paragraphs in each section
        """        
        if len(sections) != len(paragraphs) or len(sections) == 0 or len(paragraphs) == 0:
            raise ValueError("Invalid document: sections and paragraphs must have the same length and at least one element")

        self.doc_id = doc_id
        self.sections = sections
        self.paragraphs = paragraphs

    def get_section(self, section_type: str) -> t.List[str]:
        """get paragraphs corresponding to a section type

        Args:
            section_type (str): section type (see preproc.Opinion.section_order)

        Returns:
            t.List[str]: list of paragraphs in section
        """        
        try:
            return self.paragraphs[self.sections.index(section_type)]
        except ValueError:
            return []
    
    def get_paragraphs(self) -> t.List[str]:
        """get all paragraphs in the document

        Returns:
            t.List[str]: list of paragraphs
        """        
        # flatten a list of lists
        return [item for sublist in self.paragraphs for item in sublist]