import re
import pytest
from temibox.preprocessor import SimplePreprocessor


def test_single_text_inputs():

    preprocessor = SimplePreprocessor(clean_fn = None)
    out = preprocessor.process("12345")
    assert isinstance(out, list) and len(out) == 1 and len(out[0]) == 5, "Should return a one element list with a string"

    out = preprocessor.process("")
    assert isinstance(out, list) and len(out) == 1 and len(out[0]) == 0, "Should return a one element list with an empty string"


def test_multi_text_inputs():

    preprocessor = SimplePreprocessor(clean_fn=None)
    out = preprocessor.process(["12345", "54321"])
    assert isinstance(out, list) and len(out) == 2 and all([len(x) for x in out]), "Should return a two element list with a string"

@pytest.mark.parametrize("str_in,str_out,clean_fn", [
    ("BlaBlA!", r"aA", lambda x: re.sub(r"[^a]+","", x, flags=re.I)),
    ("\n   trimmed  \n", r"trimmed", lambda x: x.strip()),
    ("LOWCASE", r"lowcase", lambda x: x.lower()),
])
def test_clean_fn(str_in, str_out, clean_fn):
    preprocessor = SimplePreprocessor(clean_fn = clean_fn)
    out = preprocessor.process(str_in)

    assert isinstance(out, list) and len(out) == 1 and out[0] == str_out, "Cleaning should work as expected"


