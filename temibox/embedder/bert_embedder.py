from ..tokenizer import BertTokenizer
from ..vectorizer import BertVectorizer
from .derived_embedder import DerivedEmbedder


class BertEmbedder(DerivedEmbedder):
    r"""
    DerivedEmbedder embedder consisting of a BERT Tokenizer and
    BERT Vectorizer
    """

    def __init__(self,
                 pretrained_model_dir: str,
                 allow_max_new_tokens: int = 256,
                 min_new_token_freq:   int = 32,
                 max_sequence_length:  int = 512):

        tokenizer = BertTokenizer(pretrained_model_dir = pretrained_model_dir,
                                  allow_max_new_tokens = allow_max_new_tokens,
                                  min_new_token_freq   = min_new_token_freq,
                                  max_sequence_length  = max_sequence_length)

        vectorizer = BertVectorizer(pretrained_model_dir = pretrained_model_dir)

        super().__init__(tokenizer = tokenizer, vectorizer = vectorizer)
