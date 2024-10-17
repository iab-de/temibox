import spacy
from typing import Dict, Any, List, Callable

from .preprocessor import Preprocessor


class SpacyPreprocessor(Preprocessor):

    def __init__(self,
                 spacy_model: str = "de_core_news_lg",
                 clean_fn: Callable[[str], str] = None):

        super().__init__()
        self._nlp = spacy.load(spacy_model)
        self._clean_fn = clean_fn

    # Pipeline methods
    def train(self, **kwargs) -> None:
        pass

    def transform(self, text: str | List[str] | None, **kwargs) -> Dict[str, Any]:
        return {"text": self.process(text = text)}

    def clean(self) -> None:
        pass

    # Preprocessor methods
    def process(self, text: str | List[str]) -> List[List[str]]:

        if not isinstance(text, List):
            text = [text]

        texts = [[]] * len(text)

        for i, t in enumerate(text):
            sents = []
            for sent in self._nlp(t).sents:
                text = sent.text

                if self._clean_fn:
                    text = self._clean_fn(text)
                sents.append(text)

            texts[i] = sents

        return texts

