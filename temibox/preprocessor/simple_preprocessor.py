from typing import Dict, Any, List, Callable

from .preprocessor import Preprocessor

def _default_clean_fn(text: str) -> str:
    pass

class SimplePreprocessor(Preprocessor):

    def __init__(self, clean_fn: Callable[[str], str] | None = _default_clean_fn):
        super().__init__()
        self._clean_fn = clean_fn

    # Pipeline methods
    def train(self, **kwargs) -> None:
        pass

    def clean(self) -> None:
        pass

    def transform(self, text: str | List[str], **kwargs) -> Dict[str, Any]:
        return {"text": self.process(text=text)}

    # Preprocessor methods
    def process(self, text: str | List[str]) -> List[str]:

        if not isinstance(text, List):
            text = [text]

        ctext = [""] * len(text)
        for i, t in enumerate(text):
            if self._clean_fn:
                t = self._clean_fn(t)
            ctext[i] = t

        return ctext

