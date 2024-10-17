import random
import numpy as np
from collections import Counter

from temibox.domain import UseCase, Label, Triplet, Document

from .document import Publication


class Themenzuordnung(UseCase):

    def __init__(self):
        self._label_lookup:        dict[int, Label]       = {}
        self._label_roots:         dict[int, Label]       = {}
        self._label_sameroot:      dict[int, list[int]]   = {}
        self._label_reverse_index: dict[str, list[Label]] = {}
        self._label_weights:       dict[Label, float]     = {}

    @property
    def name(self) -> str:
        return "Themenzuordnung"

    def get_document_body(self, document: Publication) -> str:
        return ". ".join([document.title, *document.keywords, document.abstract]).replace("?.", "?").replace("!.", "!")

    def get_document_labels(self, document: Publication) -> list[Label]:
        return [label for tid in document.topics if (label := self._label_lookup.get(tid, None))]

    def get_document_label_ids(self, document: Document) -> list[int]:
        return [tid for tid in document.topics if self._label_lookup.get(tid, None)]

    def get_usecase_labels(self) -> set[Label]:
        return set(self._label_weights.keys()).copy()

    def get_usecase_label_weights(self) -> dict[Label, float]:
        return self._label_weights.copy()

    def get_usecase_label_dict(self) -> dict[int, Label]:
        return self._label_lookup.copy()

    def create_document_test_labels(self, document: Publication, positive: int, negative: int) -> list[tuple[(Label, bool)]]:
        plabels = set(self.get_document_labels(document))
        nlabels = self.get_usecase_labels() - plabels

        pos = random.choices(tuple(plabels), k = positive)
        neg = random.choices(tuple(nlabels), k = negative)

        total = [(x, True) for x in pos] + [(x, False) for x in neg]
        random.shuffle(total)

        return total

    def create_document_triplet(self, document: Publication, examples: int = 1) -> list[Triplet]:

        soft_examples = examples // 4
        hard_examples = examples - soft_examples

        positive_topic_ids = [t for t in document.topics if t in self._label_lookup]
        positive_topic_names = {self._label_lookup[tid] for tid in positive_topic_ids}

        positive_roots = {r for t in positive_topic_ids if (r := self._label_roots.get(t, None)) is not None}
        positive_words = {w.strip(".,") for t in document.topics for w in self._label_lookup.get(t, "").lower().split() if
                          len(w) > 3}

        positive_ids = list(np.random.choice(positive_topic_ids, examples, replace=True))
        negative_ids = []

        # hard negatives - same positive root
        hard_negatives = {tid for t in positive_roots for tid in self._label_sameroot[t] if
                          tid in self._label_lookup and tid not in positive_topic_ids and self._label_lookup[
                              tid] not in positive_topic_names}

        # hard word negatives - same words
        hard_word_negatives = {tid for w in positive_words for tid in self._label_reverse_index.get(w, []) if
                               tid not in positive_topic_ids and self._label_lookup[tid] not in positive_topic_names}

        hard_negatives = list(hard_negatives)
        if len(hard_negatives):
            negative_ids += list(
                np.random.choice(hard_negatives, min(len(hard_negatives), hard_examples), replace=False))

        if examples - len(negative_ids):
            soft_base = list(hard_word_negatives) if len(hard_word_negatives) else list(
                {x for x in (self._label_lookup.keys() - positive_ids) if
                 self._label_lookup[x] not in positive_topic_names})
            negative_ids += list(np.random.choice(soft_base, examples - len(negative_ids), replace=True))

        random.shuffle(negative_ids)

        body = self.get_document_body(document)
        triplets = [Triplet(positive = positive_ids[i], anchor = body, negative = negative_ids[i])
                    for i in range(examples)]

        return triplets

    # Custom methods
    def prepare(self,
                documents:           list[Publication],
                topic_lookup:        dict[int, Label],
                topic_roots:         dict[int, Label],
                topic_sameroot:      dict[int, list[int]],
                topic_reverse_index: dict[str, list[int]],
                **kwargs):

        self._label_lookup        = topic_lookup
        self._label_roots         = topic_roots
        self._label_sameroot      = topic_sameroot
        self._label_reverse_index = topic_reverse_index

        counter = Counter()
        for doc in documents:
            counter.update(set(doc.topics))

        wmax = counter.most_common(1)[0][1]
        self._label_weights = {self._label_lookup.get(k, None): v/wmax for k, v in counter.items()}


class Verschlagwortung(Themenzuordnung):

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "Verschlagwortung"

    def get_document_body(self, document: Publication) -> str:
        return ". ".join([document.title, document.abstract]).replace("?.", "?").replace("!.", "!")

    def get_document_labels(self, document: Publication) -> list[Label]:
        return document.keywords

    def get_document_label_ids(self, document: Publication) -> list[int]:
        raise Exception("TODO")

    def get_usecase_labels(self) -> set[Label]:
        return set(self._label_weights.keys())

    def get_usecase_label_weights(self) -> dict[Label, float]:
        return self._label_weights.copy()

    def create_document_triplet(self, document: Publication, examples: int = 1) -> list[Triplet]:

        plabels = set(self.get_document_labels(document))
        nlabels = self.get_usecase_labels() - plabels

        ncands = set()
        for plabel in plabels:
            cplabel = plabel.lower().strip()
            cplabel_parts = set(cplabel.split())
            for nlabel in nlabels:
                cnlabel = nlabel.lower().strip()
                cnlabel_parts = set(cnlabel.split())
                if cplabel == cnlabel:
                    break

                if cnlabel.startswith(cplabel) or cplabel.startswith(cnlabel) or (cplabel_parts & cnlabel_parts) != set():
                    ncands.add(nlabel)
                    continue

                if len(cnlabel) > 3 and len(cplabel) > 3 and cnlabel[:4] == cplabel[:4]:
                    ncands.add(nlabel)

        if len(ncands) < examples:
            ncands = nlabels

        pos = random.choices(tuple(plabels), k=examples)
        neg = random.choices(tuple(ncands),  k=examples)

        return [Triplet(positive=pos[i], anchor=self.get_document_body(document), negative=neg[i])
                for i in range(examples)]

    # Custom methods
    def prepare(self,
                documents: list[Publication], **kwargs):

        counter = Counter()
        for doc in documents:
            counter.update(set(doc.keywords))

        wmax = counter.most_common(1)[0][1]
        self._label_weights = {k: v/wmax for k,v in counter.items()}