import random
from dataclasses import dataclass
from temibox.domain import UseCase, Label, Triplet, Document


@dataclass
class Publikation:
    title:    str
    abstract: str
    topics:   list[Label]
    keywords: list[Label]
    authors:  list[Label]

class Themenzuordnung(UseCase):

    @property
    def name(self) -> str:
        return "Themenzuordnung"

    def get_document_body(self, document: Publikation) -> str:
        return document.title + ". " + document.abstract + ". " + ". ".join([str(k) for k in document.keywords])

    def get_document_labels(self, document: Publikation) -> list[Label]:
        return document.topics

    def get_usecase_labels(self) -> set[Label]:
        return {"t1", "t2", "t3", "t4", "t5"}

    def get_usecase_label_dict(self) -> dict[int, Label]:
        return {0: "t1",
                1: "t2",
                2: "t3",
                3: "t4",
                4: "t5"}

    def get_usecase_label_weights(self) -> dict[Label, float]:
        return {"t1": 0.25, "t2": 0.15, "t3": 0.10, "t4": 0.39, "t5": 0.11}

    def get_document_label_ids(self, document: Document) -> list[int]:
        return [k for k,v in self.get_usecase_label_dict().items() if v in document.topics]

    def create_document_test_labels(self, document: Publikation, positive: int, negative: int) -> list[tuple[Label, bool]]:
        plabels = set(self.get_document_labels(document))
        nlabels = self.get_usecase_labels() - plabels

        pos = random.choices(tuple(plabels), k = positive)
        neg = random.choices(tuple(nlabels), k = negative)

        total = [(x, True) for x in pos] + [(x, False) for x in neg]
        random.shuffle(total)

        return total

    def create_document_triplet(self, document: Publikation, examples: int = 1) -> list[Triplet]:
        triplets = []

        for _ in range(examples):
            plabels = set(self.get_document_labels(document))
            nlabels = self.get_usecase_labels() - plabels
            pos = random.choices(tuple(plabels), k=1)
            neg = random.choices(tuple(nlabels), k=1)

            triplets.append(Triplet(positive=pos, anchor=self.get_document_body(document), negative=neg))

        return triplets

usecase = Themenzuordnung()
doc = Publikation(title    = "This is a publication title",
                  abstract = "This is its abstract",
                  topics   = ["t1", "t5"],
                  keywords = ["a", "c"],
                  authors  = ["pytest"])

def test_general_properties():

    assert usecase.name == "Themenzuordnung", "Name is wrong"
    assert usecase.get_usecase_labels() == {"t1", "t2", "t3", "t4", "t5"}, "Labels are wrong"
    assert usecase.get_usecase_label_weights() ==  {"t1": 0.25, "t2": 0.15, "t3": 0.10, "t4": 0.39, "t5": 0.11}, "Weights are wrong"

def test_document_properties():

    assert usecase.get_document_body(doc) == "This is a publication title. This is its abstract. a. c", "Body is wrong"
    assert set(usecase.get_document_labels(doc)) == {"t1", "t5"}, "Labels are wrong"

def test_document_test_labels():
    sample = usecase.create_document_test_labels(doc, 4, 5)

    assert len(sample) == 9, "Sample size is wrong"
    assert {t for t, b in sample if b} == set(usecase.get_document_labels(doc)), "Sample positive examples are wrong"
    assert {t for t, b in sample if not b} & set(usecase.get_document_labels(doc)) == set(), "Sample negative examples are wrong"

def test_document_triplet():
    triplet = usecase.create_document_triplet(doc, examples=1)[0]

    assert set(triplet.positive) & {"t1", "t5"} != set(), "Positive triplet example is wrong"
    assert set(triplet.negative) & {"t1", "t5"} == set(), "Negative triplet example is wrong"
    assert triplet.anchor == usecase.get_document_body(doc), "Triplet anchor is wrong"

def test_inverse_weights():
    nweights = usecase.get_usecase_label_weights()
    wmin = min(nweights.values())
    for t, w in usecase.get_usecase_label_inverse_weights().items():
        assert abs(w - wmin / nweights[t]) < 1e-5, f"Inverse weight for {t} is wrong"

def test_binary_weights():

    lookup =  {"t1": 0.25, "t2": 0.15, "t3": 0.10, "t4": 0.39, "t5": 0.11}
    test_cases = [["t5", "t4", "t3", "t2", "t1"],
                  ["t5", "t3", "t1"] ,
                  ["t4", "t2"] ,
                  ["t4", "t7", "t1"],
                  ["t10", "t11", "t12", "t13"]]

    for test_labels in test_cases:
        weights = usecase.get_label_weights(test_labels)
        assert all([abs(lookup.get(label, 0.0) - weights[i]) < 1e-5 for i, label in enumerate(test_labels)]), "Wrong weights"


def test_binary_inverse_weights():

    lookup =  {"t1": 0.25, "t2": 0.15, "t3": 0.10, "t4": 0.39, "t5": 0.11}
    wmin = min(lookup.values())
    test_cases = [["t5", "t4", "t3", "t2", "t1"],
                  ["t5", "t3", "t1"] ,
                  ["t4", "t2"] ,
                  ["t4", "t7", "t1"],
                  ["t10", "t11", "t12", "t13"]]

    for test_labels in test_cases:
        weights = usecase.get_label_inverse_weights(test_labels)
        assert all([abs(wmin/lookup.get(label, 1.0) - weights[i]) < 1e-5 for i, label in enumerate(test_labels)]), "Wrong weights"

def test_multinomial_indicators():

    labels = sorted(list(usecase.get_usecase_labels()))
    for i in range(2**len(labels)):
        idx = [float(x) for x in reversed(f"{i:010b}")][:len(labels)]
        perm_labels = [labels[i] for i, x in enumerate(idx) if int(x) == 1]

        d = Publikation(title="", abstract="", topics=perm_labels, keywords=[], authors=[])

        assert usecase.get_multinomial_indicators(d) == idx, "Multinomial indicators are wrong"

def test_weights():

    w = usecase.get_usecase_label_weights()
    assert abs(sum(w.values()) - 1 ) < 1e-5, "Sum of weights not close to 1"

    iw = usecase.get_usecase_label_inverse_weights()

    assert abs(iw["t1"] - min(w.values()) / w["t1"]) < 1e-5, "Inverse weight for t1 wrong (1)"
    assert abs(iw["t2"] - min(w.values()) / w["t2"]) < 1e-5, "Inverse weight for t2 wrong (1)"
    assert abs(iw["t3"] - min(w.values()) / w["t3"]) < 1e-5, "Inverse weight for t3 wrong (1)"
    assert abs(iw["t4"] - min(w.values()) / w["t4"]) < 1e-5, "Inverse weight for t4 wrong (1)"
    assert abs(iw["t5"] - min(w.values()) / w["t5"]) < 1e-5, "Inverse weight for t5 wrong (1)"

    assert abs(usecase.get_label_inverse_weights(["t1"])[0] - min(w.values()) / w["t1"]) < 1e-5, "Inverse weight for t1 wrong (2)"
    assert abs(usecase.get_label_inverse_weights(["t2"])[0] - min(w.values()) / w["t2"]) < 1e-5, "Inverse weight for t2 wrong (2)"
    assert abs(usecase.get_label_inverse_weights(["t3"])[0] - min(w.values()) / w["t3"]) < 1e-5, "Inverse weight for t3 wrong (2)"
    assert abs(usecase.get_label_inverse_weights(["t4"])[0] - min(w.values()) / w["t4"]) < 1e-5, "Inverse weight for t4 wrong (2)"
    assert abs(usecase.get_label_inverse_weights(["t5"])[0] - min(w.values()) / w["t5"]) < 1e-5, "Inverse weight for t5 wrong (2)"
