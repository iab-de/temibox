from temibox.preprocessor import SpacyPreprocessor

def test_single_text_inputs():
    preprocessor = SpacyPreprocessor(spacy_model = "de_core_news_lg")

    out = preprocessor.process(text = "This is a test. Testing the spacy preprocessor.")
    assert len(out) == 1 and len(out[0]) == 2, "Should return a list containing a list with two sentences"

    out = preprocessor.process(text="")
    assert len(out) == 1 and len(out[0]) == 0, "Should return a list containing an empty list"


def test_multi_text_inputs():
    preprocessor = SpacyPreprocessor(spacy_model = "de_core_news_lg")

    out = preprocessor.process(text = ["This is a test. Testing the spacy preprocessor.",
                                       "This is another text. I expect three sentences. Three."])

    assert len(out) == 2 and len(out[0]) == 2 and len(out[1]) == 3, "Should return a two element list with lists containing two and three elements"

