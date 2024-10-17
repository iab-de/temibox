import os
import ast
import random
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
from typing import Optional, Any, Callable

from temibox.context import Context
from temibox.traits import Trainable, Transformable, Cleanable, Evaluating
from temibox.domain import Document

# Local domain
from .document import Publication
from .usecases import Themenzuordnung, Verschlagwortung


class DatenLader(Trainable, Transformable, Evaluating, Cleanable):

    def __init__(self,
                 clean_fn: Callable[[str], str] = None,
                 max_pub_topics:   int = 10,
                 max_pub_keywords: int = 20):

        super().__init__()

        self._clean_fn            = clean_fn if clean_fn is not None else self._default_clean_fn
        self._max_pub_topics      = max_pub_topics
        self._max_pub_keywords    = max_pub_keywords

        self._topic_lookup        = {}
        self._topic_roots         = {}
        self._topic_sameroot      = {}
        self._topic_reverse_index = {}

        self._pubs = []
        self._stopwords: set[str] = set()

    def _default_clean_fn(self, x: str) -> str:
        return x

    def _load_df_pubs(self,
                   load_from_file: str,
                   max_docs: int = -1,
                   skiprows: int = -1,) -> DataFrame:

        if max_docs > 0:
            df = pd.read_csv(load_from_file, sep=";", decimal=",", encoding="utf-8", nrows=max_docs, skiprows=max(skiprows, 0))
        else:
            df = pd.read_csv(load_from_file, sep=";", decimal=",", encoding="utf-8", skiprows=max(skiprows, 0))

        return df

    def _create_pub(self,
                    pub_id: int,
                    title: str,
                    abstract: str,
                    keywords: list[str],
                    topics: list[int],
                    language: str = "unk"):

        clean_title    = self._clean_fn(title)
        clean_abstract = self._clean_fn(abstract)

        pub = Publication(pub_id       = pub_id,
                          title        = clean_title,
                          abstract     = clean_abstract,
                          topics       = topics,
                          keywords     = keywords,
                          language     = language)

        return pub

    def _get_fulltext(self, title: str, abstract: str, keywords: list[str], clean: bool = True) -> str:

        if clean:
            clean_title, clean_abstract = self._clean_fn(title), self._clean_fn(abstract)
        else:
            clean_title, clean_abstract = title, abstract

        return ". ".join([clean_title, *keywords, clean_abstract])

    def _process_topics(self,
                        df_topics:           DataFrame,
                        use_topic_metadata:  bool,
                        use_full_topic_path: bool,
                        bad_topic_prefixes:  list[str] = None,
                        stopwords:           set[str]  = None):

        topic_lookup        = {}
        topic_roots         = {}
        topic_sameroot      = {}
        topic_reverse_index = {}
        stopwords           = stopwords or set()

        for _, t in df_topics.query("ungueltig == False").iterrows():

            themen_pfad = ast.literal_eval(t.themen_pfad)
            themen_id_pfad = ast.literal_eval(t.themen_id_pfad)

            if use_topic_metadata:
                thema = ". ".join(themen_pfad)
                keywords = [str(k) for k in ast.literal_eval(t.schlagwoerter) if k is not None]
                description = str(t.beschreibung)
            else:

                if use_full_topic_path:
                    thema = ", ".join(themen_pfad)
                else:
                    thema = ", ".join([themen_pfad[0], themen_pfad[-1]]) if len(themen_pfad) >= 2 else themen_pfad[0]

                keywords = []
                description = ""

            if bad_topic_prefixes is not None:
                cthema = thema.lower().strip()

                for p in bad_topic_prefixes:
                    if cthema.startswith(p.lower()):
                        thema = None
                        break

            if thema is None:
                continue

            # Topic root
            topic_root = themen_id_pfad[0]
            topic_id   = themen_id_pfad[-1]
            if topic_root not in topic_sameroot:
                topic_sameroot[topic_root] = set()

            topic_roots[topic_id] = topic_root
            topic_sameroot[topic_root].update(set(themen_id_pfad))

            # Topic words
            for word in set(self._clean_fn(themen_pfad[-1]).lower().split()):
                if len(word) < 4 or word in stopwords:
                    continue

                if word not in topic_reverse_index:
                    topic_reverse_index[word] = set()

                topic_reverse_index[word].add(topic_id)

            topic_lookup[topic_id] = self._get_fulltext(thema, description, keywords)

        print(f"Prepared {len(topic_lookup)} topics with {len(topic_sameroot)} roots")

        return topic_lookup, topic_roots, topic_sameroot, topic_reverse_index

    def _process_pubs(self,
                      df_pubs:             DataFrame,
                      df_feedback:         DataFrame = None,
                      permitted_types:     list[str] = None,
                      permitted_languages: list[str] = None,
                      duplication_matrix:  DataFrame = None) -> list[Publication]:

        if permitted_types is not None:
            print(f"Permitted Publication types: {', '.join(permitted_types)}")
            df_pubs = df_pubs.loc[df_pubs.typ.isin(permitted_types),].reset_index(drop=True)

        if permitted_languages is not None:
            print(f"Permitted Publication languages: {', '.join(permitted_languages)}")
            df_pubs = df_pubs.loc[df_pubs.language.isin(permitted_languages),].reset_index(drop=True)

        if duplication_matrix is None:
            duplication_matrix = DataFrame([{"title": True, "abstract": True, "keywords": True}])

        pubs = []
        languages = {}
        for _, row in tqdm(df_pubs.iterrows(), "Loading Publications"):

            raw_topics = set(ast.literal_eval(row["topics"]))

            if df_feedback is not None:
                feedback     = df_feedback.query(f"bibdia_id == {row.pub_id}")
                feedback_pos = set(feedback.query("positive == 1").topic_id.tolist())
                feedback_neg = set(feedback.query("positive == 0").topic_id.tolist())

                raw_topics |= feedback_pos - feedback_neg

            pub_id   = row["pub_id"]
            title    = self._clean_fn(row["title"])
            abstract = self._clean_fn(row["abstract"])
            keywords = [keyword for k in ast.literal_eval(row["keywords"]) if (keyword := self._clean_fn(k))]
            topics   = [tid for tid in raw_topics if tid in self._topic_lookup]
            language = row["language"]

            if not len(topics) or len(topics) > self._max_pub_topics:
                continue

            languages[language] = languages.get(language, 0) + 1

            for _, drow in duplication_matrix.iterrows():

                dup_title = ""
                dup_abstract = ""
                dup_keywords = []

                if drow.title:
                    dup_title = title

                if drow.keywords:
                    dup_keywords = keywords

                if drow.abstract:
                    dup_abstract = abstract

                if len(self._get_fulltext(title, abstract, keywords)) > 100:
                    pub = self._create_pub(pub_id   = pub_id,
                                           title    = dup_title,
                                           abstract = dup_abstract,
                                           keywords = dup_keywords,
                                           topics   = topics,
                                           language = language)
                    pubs.append(pub)

        print(f"Loaded {len(pubs)} Publications. Sprachen: " + ", ".join(
            [f"{k}: {v / sum(languages.values()):.2%}" for k, v in languages.items()]))

        random.shuffle(pubs)

        return pubs

    def train(self,
              ctx:                 Optional[Context],
              publication_df_path: str       = None,
              topic_df_path:       str       = None,
              feedback_df_path:    str       = None,
              stopword_text_path:  str       = None,
              max_docs:            int       = -1,
              skiprows:            int       = -1,
              duplication_matrix:  DataFrame = None,
              permitted_types:     list[str] = None,
              permitted_languages: list[str] = None,
              use_full_topic_path: bool      = False,
              use_topic_metadata:  bool      = False,
              bad_topic_prefixes:  list[str] = None,
              **kwargs) -> None:

        self._pubs = []
        self._stopwords = set()

        self._topic_lookup        = {}
        self._topic_roots         = {}
        self._topic_sameroot      = {}
        self._topic_reverse_index = {}

        # Raw data
        df_pubs      = self._load_df_pubs(publication_df_path, max_docs, skiprows)

        df_topics = pd.read_csv(topic_df_path, sep=";", decimal=",", encoding="utf-8")

        if feedback_df_path:
            df_feedback = pd.read_csv(feedback_df_path, sep=";", decimal=",", encoding="utf-8")
        else:
            df_feedback = None

        # Stopwords
        if stopword_text_path:
            with open(stopword_text_path, "r") as f:
                self._stopwords = set(f.read().split())

        # Topics
        self._topic_lookup, \
        self._topic_roots, \
        self._topic_sameroot, \
        self._topic_reverse_index = self._process_topics(df_topics           = df_topics,
                                                         use_topic_metadata  = use_topic_metadata,
                                                         use_full_topic_path = use_full_topic_path,
                                                         bad_topic_prefixes  = bad_topic_prefixes,
                                                         stopwords           = self._stopwords)

        # Publications
        self._pubs = self._process_pubs(df_pubs             = df_pubs,
                                        df_feedback         = df_feedback,
                                        permitted_types     = permitted_types,
                                        permitted_languages = permitted_languages,
                                        duplication_matrix  = duplication_matrix)

        # Pass documents to usecases
        for usecase in ctx.usecases:
            if usecase.name in self.registered_usecase_names and (isinstance(usecase, Themenzuordnung) or isinstance(usecase, Verschlagwortung)):
                usecase.prepare(**self._get_data(ctx))


    def _get_data(self, ctx: Context, documents: list[Document] = None):
        active_usecase_name = ctx.active_usecase.name if ctx.active_usecase else None

        return {"label_dict":          self._topic_lookup.copy() if active_usecase_name == "Themenzuordnung" else None,

                "documents":           (documents or self._pubs).copy(),
                "topic_lookup":        self._topic_lookup.copy(),
                "topic_roots":         self._topic_roots.copy(),
                "topic_sameroot":      self._topic_sameroot.copy(),
                "topic_reverse_index": self._topic_reverse_index.copy()}

    def transform(self,
                  ctx: Optional[Context],
                  pub_id: int = 0,
                  title: str = None,
                  abstract: str = None,
                  keywords: list[str] = None,
                  **kwargs) -> dict[str, Any]:

        data = self._get_data(ctx)

        if title or abstract or keywords:
            document = self._create_pub(pub_id   = pub_id,
                                        title    = self._clean_fn(title),
                                        abstract = self._clean_fn(" ".join([x.strip() for x in abstract.split()])),
                                        keywords = [self._clean_fn(k) for k in keywords],
                                        topics   = [],
                                        language = "unk")

            data["document"] = document

        return {**data, **kwargs}

    def evaluate(self,
                 ctx: Optional[Context],
                 publication_df_path: str = None,
                 documents:           list[Document] = None,
                 max_docs:            int = 1000,
                 permitted_types:     list[str] = None,
                 permitted_languages: list[str] = None,
                 **kwargs):

        # Only evaluate after training
        if ctx.pipeline.is_training:
            return {}

        if documents is not None:
            return {"documents": documents}

        if publication_df_path is None or not os.path.isfile(publication_df_path):
            raise Exception("Evaluation dataset does not exist")

        df_pubs_test = self._load_df_pubs(publication_df_path, max_docs, -1)

        if not df_pubs_test.shape[0]:
            raise Exception("Evaluation dataset is empty")

        pubs_test = None
        if df_pubs_test is not None:
            pubs_test = self._process_pubs(df_pubs=df_pubs_test,
                                           df_feedback=None,
                                           permitted_types=permitted_types,
                                           permitted_languages=permitted_languages,
                                           duplication_matrix=None)

        return {"documents": pubs_test}

    def clean(self) -> None:
        self._pubs = []
