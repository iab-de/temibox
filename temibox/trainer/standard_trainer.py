import torch
import random
import numpy as np
from enum import Enum
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Generator, Any, Optional

from .trainer import Trainer
from ..traits import Supervising, PipelineStep, Cleanable
from ..capabilities import ParameterCapable, InferenceCapable, CudaCapable
from ..vectorizer.vectorizer import Vectorizer
from ..embedder.embedder import Embedder
from ..model.supervised_model import SupervisedModel
from ..pipeline.pipeline import Pipeline
from ..context import Context, ContextArg
from ..domain import Document


class StandardTrainer(Trainer, Supervising, Cleanable):

    @dataclass
    class Trainplan:
        epochs: int
        learn_rate: float
        batch_size: int
        freeze_vectorizer: bool
        max_docs: int = -1
        break_early: bool = False
        reset_break_early_score: bool = False

    @dataclass
    class _LossHistory:
        train: list[float]
        test:  list[float]

    class RUNTYPE(Enum):
        TRAIN = "train"
        TEST = "test"

    def __init__(self,
                 train_test_split: float = 0.9,
                 max_no_improvement: int = 2,
                 create_checkpoints: bool = True):

        super().__init__()

        self._train_split = min(1.0, max(0.0, train_test_split))

        self._supervised: list[Any] = []
        self._supervised_vectorizers: dict[str, Vectorizer | Embedder] = {}
        self._supervised_models: dict[str, SupervisedModel] = {}

        self._loss_history: list[StandardTrainer._LossHistory] = []
        self._train_phase = 0
        self._create_checkpoints = create_checkpoints

        self._best_score = np.inf
        self._best_epoch = -1
        self._no_improvement = 0
        self._max_no_improvement = max_no_improvement
        self._checkpoint_items = {}

    def _get_batchgen(self, docs: list[Document], batch_size: int) -> Generator[list[Document], None, None]:

        docs = docs.copy()
        random.shuffle(docs)

        batch: list[Document] = []
        for doc in docs:
            batch.append(doc)

            if len(batch) == batch_size:
                yield batch
                batch = []

        if len(batch):
            yield batch

    def _print_loss_history(self, title: str, values: list[float]):
        q = [0.5, 0.75, 0.95]
        v = np.quantile(values, q)
        qv = ", ".join([f"Q{q[i]*100:.0f}: {v[i]:.4f}" for i in range(len(q))])

        print(f"\n{title}: {np.mean(values):.4f} ({qv})")

    def _backprop(self, optimizer, losses: list[torch.Tensor]):

        optimizer.zero_grad()
        loss = sum(losses)
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

            for supervised in self._supervised:
                if isinstance(supervised, InferenceCapable):
                    supervised.zero_grad()

        return loss.detach().item()

    def _run_epoch(self,
                   ctx: Context,
                   run_type: RUNTYPE,
                   plan: Trainplan,
                   docs: list[Document],
                   history: list[float]):

        epoch = len(self._loss_history) + 1

        # Enable embedder cache and toggle inference mode
        for embedder in self._supervised_vectorizers.values():
            embedder.configure_cache(on = True, max_entries = 1024)

            if isinstance(embedder, InferenceCapable):
                embedder.set_inference_mode(on = plan.freeze_vectorizer)

        params = [p for m in self._supervised_models.values() if isinstance(m, ParameterCapable) for p in m.get_training_parameters()]
        if not plan.freeze_vectorizer:
            params += [p for e in self._supervised_vectorizers.values() if isinstance(e, ParameterCapable) for p in e.get_training_parameters()]

        if not len(params):
            print("No trainable parameters found, skipping")
            return

        optimizer = torch.optim.Adam(params = params, lr = plan.learn_rate)

        for batch in tqdm(self._get_batchgen(docs, plan.batch_size), "Trainer loop"):

            # Calculate losses
            losses = []
            for model in self._supervised_models.values():
              losses += model.get_losses(ctx = ctx, documents = batch)

            # Backprop
            loss = self._backprop(optimizer, losses)

            # Clear cache
            for embedder in self._supervised_vectorizers.values():
                embedder.clear_cache()

            # Store / display history
            history.append(loss)
            if run_type == self.RUNTYPE.TRAIN and len(history) % 32 == 0:
                self._print_loss_history(f"[p{self._train_phase}::e{epoch}::{run_type}]", history[-32:])

        # Print total loss
        self._print_loss_history(f"[p{self._train_phase}::e{epoch}::{run_type}]", history)

        # Disable embedder cache
        for embedder in self._supervised_vectorizers.values():
            embedder.configure_cache(on = False)

        # Checkpoint
        if run_type == self.RUNTYPE.TEST:
            if plan.break_early:
                self._checkpoint(ctx.pipeline, np.mean(history).item())
            else:
                self._no_improvement = 0

    def _get_default_trainplans(self) -> list['StandardTrainer.Trainplan']:

        return [
                self.Trainplan(epochs=1,
                               learn_rate=1e-3,
                               batch_size=8,
                               max_docs = 50_000,
                               freeze_vectorizer=True,
                               break_early=False,
                               reset_break_early_score=False),

                self.Trainplan(epochs=10,
                               learn_rate=1e-5,
                               batch_size=8,
                               freeze_vectorizer=False,
                               break_early = True,
                               reset_break_early_score = False),

                self.Trainplan(epochs=10,
                               learn_rate=1e-6,
                               batch_size=8,
                               freeze_vectorizer=False,
                               break_early = True,
                               reset_break_early_score = False)
                ]

    def _checkpoint(self, pipeline: Pipeline, mean_loss: float):

        if mean_loss < self._best_score:
            if self._best_score < np.inf:
                print(f"Score {mean_loss:.4f} is better than the old best score of {self._best_score:.4f}")
            else:
                print(f"Best score - {mean_loss:.4f}")

            self._best_score = mean_loss
            self._no_improvement = 0
            self._best_epoch = len(self._loss_history) + 1
            self._checkpoint_items = {}

            if self._create_checkpoints:
                copy = pipeline.deepcopy()
                for name in self._supervised_vectorizers.keys() | self._supervised_models.keys():
                    self._checkpoint_items[name] = copy.get_step(name)

                del copy
        else:
            if self._no_improvement > 0:
                print(f"No improvement ({self._no_improvement+1}/{self._max_no_improvement+1}): new score {mean_loss:.4f} is worse than the previous best score of {self._best_score:.4f}")
            else:
                print(f"No improvement: new score {mean_loss:.4f} is worse than the previous best score of {self._best_score:.4f}")

            self._no_improvement += 1

    def _restore(self, pipeline: Pipeline):

        if not self._create_checkpoints or self._no_improvement == 0 or len(self._checkpoint_items) == 0:
            return

        print(f"Restoring best checkpoint from epoch {self._best_epoch}")
        for name, step in self._checkpoint_items.items():

            old_step = pipeline.get_step(name)

            if isinstance(old_step, CudaCapable):
                step.set_cuda_mode(on=old_step.is_cuda)

            if isinstance(old_step, InferenceCapable):
                step.set_inference_mode(on=old_step.is_inference)

            pipeline.replace_step(name, step)

        self._no_improvement = 0

    def train(self,
              ctx: Optional[Context] = None,
              documents: list[Document] | ContextArg[list[Document]] = None,
              trainplans: list[Trainplan] | ContextArg[list[Trainplan]] = None,
              max_epochs: int = -1,
              **kwargs) -> None:

        if documents is None:
            raise Exception("No documents provided")

        if trainplans is None:
            trainplans = self._get_default_trainplans()
        else:
            trainplans = ContextArg.extract(trainplans)

        self._train_phase += 1

        docs = ContextArg.extract(documents).copy()
        random.shuffle(docs)

        if self._train_split > 0.999:
            print("Training will not have a TEST/VALIDATION phase")
            train_docs = docs
            test_docs = None
        else:
            tid = int(self._train_split * len(docs))
            train_docs: list[Document] = docs[:tid]
            test_docs: list[Document]  = docs[tid:]

        if not self._create_checkpoints or self._train_split > 0.999:
            print("Training will not create checkpoints")

        with ctx.pipeline.modes(inference = False, cuda = True):

            for plan in trainplans:

                if plan.reset_break_early_score:
                    self._no_improvement = 0
                    self._best_score     = np.inf

                for _ in range(plan.epochs):

                    if 0 < max_epochs <= len(self._loss_history):
                        break

                    plan_history = self._LossHistory(train = [], test=[])

                    # Train
                    self._run_epoch(ctx,
                                    run_type = self.RUNTYPE.TRAIN,
                                    plan = plan,
                                    docs = train_docs if plan.max_docs <= 0 else train_docs[:plan.max_docs],
                                    history = plan_history.train)

                    # Test
                    if test_docs is not None and len(test_docs) > 0:
                        with torch.no_grad():
                            self._run_epoch(ctx,
                                            run_type = self.RUNTYPE.TEST,
                                            plan = plan,
                                            docs = test_docs,
                                            history = plan_history.test)

                    self._loss_history.append(plan_history)

                    # Break early
                    if plan.break_early and self._no_improvement > self._max_no_improvement:
                        print("Breaking early")
                        self._restore(ctx.pipeline)
                        break

            # Restore best run
            self._restore(ctx.pipeline)

    def plot_history(self):

        if len(self._loss_history) < 2:
            print("No history to plot")
            return

        train = [np.mean(self._loss_history[0].train[:8])] + [np.mean(h.train) for h in self._loss_history]
        test  = [None]  + [np.mean(h.test) for h in self._loss_history]
        x     = [int(i) for i in range(0,len(train))]

        fig = plt.figure()
        plt.plot(x, train, label="train", figure=fig)

        if len(test):
            plt.plot(x, test, label="test", figure=fig)

        if self._create_checkpoints and self._best_epoch > 0:
            plt.vlines(self._best_epoch, min(train + test[1:])*0.99, max(train + test[1:]), linestyles = "dashed", colors="black", label="checkpoint", figure=fig)

        plt.legend(loc="upper right")

        return {"plot": fig,
                "data": self._loss_history.copy(),
                "best_score": self._best_score,
                "best_epoch": self._best_epoch}

    def supervise(self, name: str, step: PipelineStep, **kwargs) -> None:

        self._supervised.append(step)

        if isinstance(step, Vectorizer) or isinstance(step, Embedder):
            self._supervised_vectorizers[name] = step

        if isinstance(step, SupervisedModel):
            self._supervised_models[name] = step

    def supervising(self) -> list[str]:
        return list(set(self._supervised_vectorizers.keys()) | self._supervised_models.keys())

    def clean(self) -> None:
        self._loss_history = []