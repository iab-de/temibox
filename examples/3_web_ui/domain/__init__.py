import os
import re
import torch
from datetime import datetime
from pandas import DataFrame
from typing import Optional, Any

from temibox.context import Context
from temibox.pipeline import StandardPipeline
from temibox.tokenizer import BertTokenizer
from temibox.vectorizer import BertVectorizer
from temibox.traits import Predictable
from temibox.context import ContextArg
from temibox.prediction import Prediction

PRETRAINED_DIR = os.getenv("PRETRAINED_DIR")

from .usecase import WebUseCase, WebDocument

class NaiveKeywordExtractor(Predictable):

    def use_identifier(self, identifier: str):
        pass

    def predict(self,
                ctx: Optional[Context] = None,
                document: WebDocument = None,
                tokenizer: ContextArg[BertTokenizer] = None,
                vectorizer: ContextArg[BertVectorizer] = None,
                tokens: ContextArg[dict[str, Any]] = None,
                **kwargs) -> list[Prediction]:

        tokenizer  = ContextArg.extract(tokenizer)
        vectorizer = ContextArg.extract(vectorizer)
        tokens     = ContextArg.extract(tokens)

        embeddings = vectorizer.vectorize(tokens = tokenizer.tokenize(document.title))
        token_embeddings = vectorizer.vectorize(tokens = tokens, return_mean_embedding=False).squeeze(0)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        sims = cos(embeddings, token_embeddings)

        sim_ids = sims.argsort(descending=True)
        top_token_ids = tokens["input_ids"][0,sim_ids]
        top_tokens_100 = [(x, sims[sim_id].item()) for sim_id, x in zip(sim_ids, tokenizer._tokenizer.convert_ids_to_tokens(top_token_ids))
                                 if len(x) > 4 and re.match("^[A-ZÜÄÖ]", x)]

        seen = set()
        top_tokens = []
        for x in top_tokens_100:
            if x[0] in seen:
                continue

            if x[1] < 0.5:
                continue

            top_tokens.append(x)
            seen.add(x[0])

            if len(top_tokens) == 10:
                break

        return [Prediction(
                            usecase_name = ctx.active_usecase.name,
                            model        = "NaiveKeywordExtractor",
                            timestamp    = int(datetime.now().timestamp()),
                            payload_raw  = [],
                            payload =  DataFrame({"word": [x[0] for x in top_tokens],
                                                  "score": [x[1] for x in top_tokens]}))]

pipeline = (
    StandardPipeline()
        .add_usecase(WebUseCase())
        .add_step("tokenizer", BertTokenizer(pretrained_model_dir=PRETRAINED_DIR))
        .add_step("vectorizer", BertVectorizer(pretrained_model_dir=PRETRAINED_DIR))
        .add_step("model", NaiveKeywordExtractor())
)

pipeline.train(documents = []) # nothing to train

pipeline.export(folder=f"{os.getcwd()}/examples/3_web_ui/export",
                suffix="webmodel",
                prune=True)