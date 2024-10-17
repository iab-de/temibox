from dataclasses import dataclass
from temibox.domain import UseCase, Triplet, Label

@dataclass
class WebDocument:
    title: str
    text: str

class WebUseCase(UseCase):

    @property
    def name(self) -> str:
        return "web-usecase"

    def get_document_body(self, document: WebDocument) -> str:
        return document.text

    def get_document_labels(self, document: WebDocument) -> list[Label]:
        return []

    def get_document_label_ids(self, document: WebDocument) -> list[int]:
        return []

    def get_usecase_labels(self) -> list[Label]:
        return []

    def get_usecase_label_dict(self) -> dict[int, Label]:
        return []

    def get_usecase_label_weights(self) -> dict[Label, float]:
        return {}

    def create_document_test_labels(self, document: WebDocument, positive: int, negative: int) -> list[tuple[(Label, bool)]]:
        return []

    def create_document_triplet(self, document: WebDocument, examples: int = 1) -> list[Triplet]:
        return []