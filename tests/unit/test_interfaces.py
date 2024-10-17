import inspect
from temibox.interfaces import *


_abilities =  [Trainable, Transformable, Trackable, Predictable, Supervising,
               CudaCapable, InferenceCapable, ParameterCapable]

_interfaces = [Preprocessor, Tokenizer, Vectorizer, Embedder,
               Trainer,
               Classifier, Searcher, Summarizer]

def test_interfaces_are_abstract():

    for t in _abilities + _interfaces:
        assert inspect.isabstract(t), f"{t.__name__} should be abstract"

def test_interfaces_are_trainable():

    for t in _interfaces:
        assert issubclass(t, Trainable), f"{t.__name__} should be trainable"
