import os
import sys
sys.path.append(f"{os.getcwd()}/examples/1_themenzuordnung")

from temibox.context import ContextArg
from temibox.pipeline import StandardPipeline
from temibox.embedder import BertEmbedder, TFIDFEmbedder
from temibox.model.classifier import BinaryClassifier, KnnClassifier, KMeansCluster
from temibox.trainer import StandardTrainer
from temibox.evaluation import Evaluator
from temibox.evaluation.metric import PrecisionAtK
from temibox.losses import MultilabelBinaryLoss

from domain import Themenzuordnung, DatenLader, Publication

#################################
# Data
#################################
PRETRAINED_DIR = os.getenv("PRETRAINED_DIR")
DATA_DIR = os.getenv("DATA_DIR")

pub_train_df_path = f"{DATA_DIR}/themen/df_all_train_de_en.csv"
pub_test_df_path  = f"{DATA_DIR}/themen/df_all_test_de_en.csv"
topic_df_path     = f"{DATA_DIR}/themen/df_themen.csv"

#################################
# Definitions
#################################
model_1 = BinaryClassifier(multilabel = True,
                           layer_dimensions=[32],
                           loss_functions=[MultilabelBinaryLoss(positive_examples=4,
                                                                negative_examples=4,
                                                                use_class_weights=False)])

model_2 = BinaryClassifier(multilabel = True,
                           layer_dimensions=[32],
                           loss_functions=[MultilabelBinaryLoss(positive_examples=4,
                                                                negative_examples=4,
                                                                use_class_weights=False)])

model_3 = KnnClassifier(k_neighbours = 8,
                        bias_score   = 0.5,
                        bias_terms   = 1,
                        max_lookup_documents = 10_000)

model_4 = KMeansCluster(min_clusters = 32,
                        max_clusters = 512,
                        cluster_step = 64,
                        max_lookup_documents = 10_000)

bert_embedder  = BertEmbedder(pretrained_model_dir = PRETRAINED_DIR)
tfidf_embedder = TFIDFEmbedder(max_tokens = 10_000, embedding_dim = 128)

pipeline = (StandardPipeline(allow_cuda=True)

            # Domain-specific components
            .add_usecase(Themenzuordnung())
            .add_step("data-loader",      DatenLader())

            # BERT-based components
            .add_step("bert-embedder",    bert_embedder, dependencies=["data-loader"])
            .add_step("bert-model",       model_1, dependencies=["bert-embedder"])

            # TF-IDF based components
            .add_step("tfidf-embedder",   tfidf_embedder, dependencies=["data-loader"])
            .add_step("tfidf-model",      model_2, dependencies=["tfidf-embedder"])

            # Train both embedders and binary classifiers
            .add_step("trainer",          StandardTrainer())

            # Train kNN and kMeans based on the fine-tuned bert-embedder
            .add_step("knn-model",        model_3, dependencies=["data-loader", "bert-embedder", "trainer"])
            .add_step("kmeans-model",     model_4, dependencies=["data-loader", "bert-embedder", "trainer"])

            # Evaluate prediction performance
            .add_step("evaluator",        Evaluator(metrics = [PrecisionAtK(5, 0.25), PrecisionAtK(10, 0.25)]), dependencies=["data-loader", "bert-model", "tfidf-model", "knn-model", "kmeans-model"]))

#################################
# Visualize workflows
#################################

pipeline.show_workflow("train")
pipeline.show_workflow("predict")
pipeline.show_workflow("evaluate")

#################################
# Training
#################################
pipeline.train(publication_df_path   = pub_train_df_path,
               topic_df_path         = topic_df_path,
               max_docs              = 1_000,
               max_epochs            = 1,
               permitted_languages   = ["de", "en"],
               permitted_types       = ["publikation"])

pipeline.set_inference_mode(on = True, cuda = True)
pipeline.configure_cache(on = True, max_entries = 10_240)

#################################
# Evaluation
#################################
results = pipeline.evaluate(publication_df_path = pub_test_df_path,
                            permitted_languages = ["de", "en"],
                            permitted_types = ["publikation"],
                            max_docs   = 10_000,
                            min_score  = 0.15,
                            batch_size = 256)

import json

for mname in ["bert-model", "tfidf-model", "knn-model", "kmeans-model"]:
    print(mname)
    print(json.dumps(ContextArg.extract(results["evaluator"]["metrics"])[mname], indent=2))

#################################
# Export
#################################

pipeline.export(suffix="example_1", folder="examples", prune=True)


#################################
# Load and predict
#################################

del pipeline # delete existing pipeline

pipeline_loaded = StandardPipeline.load(suffix="example_1", folder="examples")
pipeline_loaded.set_inference_mode(on = True, cuda = True)

predictions = pipeline_loaded.predict(document = Publication(pub_id   = None,
                                                             title    = "Arbeitslosenquote in Bayern steigt!",
                                                             abstract = "Arbeitslosenquote in Bayern stieg Jahr-zu-Jahr um 2 Prozentpunkte",
                                                             keywords = ["Arbeitslosenquote", "Bayern"],
                                                             topics   = [],
                                                             language = "de"))

assert len(predictions) == 4, "Each model generates a prediction"
print(predictions[0].payload)
