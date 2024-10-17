import os
import json
import pickle
import logging
from io import BytesIO
from datetime import datetime
from typing import Type, Optional
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod
from typing import TypeVar, IO

from ..metadata import PipelineMetadata
from ..traits import PipelineStep, Trainable, Transformable, Cacheable, Cleanable, Predictable, Trackable, Evaluating
from ..domain import UseCase

PipelineType = TypeVar('PipelineType', bound = 'Pipeline')
StepType     = TypeVar('StepType', bound = PipelineStep)

class Pipeline(Trainable, Transformable, Predictable, Cleanable, Cacheable, Trackable, Evaluating, metaclass=ABCMeta):
    r"""Main pipeline execution interface

    This interface describes the minimum set of methods needed to execute a pipeline
    workflow
    """

    _logger = logging.getLogger("Pipeline")

    @abstractmethod
    def add_usecase(self, usecase: UseCase) -> 'Pipeline':
        r"""
        Adds a usecase to the pipeline.

        The names (UseCase.name) of the added usecases must be unique

        :param usecase: usecase to be added
        :return: self (for method chaining)
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def get_usecases(self) -> list[UseCase]:
        r"""
        Returns all the added usecases

        :return: list of usecases
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def get_step_usecases(self, name: str) -> list[UseCase]:
        r"""
        Returns the usecases that were registered with the step

        :param name: name of the step

        :return: list of usecases
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def add_step(self, name: str, step: PipelineStep, usecases: list[UseCase] = None, dependencies: list[str] = None) -> 'Pipeline':
        r""" Add a single pipeline step

        :param name:
        :param step:
        :param usecases: list of relevant usecases (will use all usecases if None provided)
        :param dependencies:
        :return: self (for method chaining)
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def replace_step(self, name: str, step: PipelineStep) -> 'Pipeline':
        r"""
        Replaces a step in the DAG

        :param name: name of the step
        :param step: new instance of the step

        :return: self
        """
        raise NotImplementedError("interface property not implemented")

    @abstractmethod
    def get_steps(self) -> dict[Type[PipelineStep], list[tuple[str, PipelineStep]]]:
        r"""
        Returns a dictionary of pipeline steps by step type

        :return: list of tuples (name, step)
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def get_step(self, name: str) -> Optional[StepType]:
        r"""
        Returns a single pipeline step

        :param name: name of the step

        :return: step (if it exists)
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def get_step_dependencies(self, name: str) -> list[StepType]:
        r"""
        Returns the step dependencies

        :param name: name of the step

        :return: list of steps (possibly empty)
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    def set_inference_mode(self, on: bool, cuda: bool) -> 'Pipeline':
        r"""
        Turns inference and CUDA modes on or off

        :param on: turn inference mode if True
        :param cuda: copy relevant steps to the GPU if True

        :return: self
        """
        raise NotImplementedError("interface method not implemented")

    @abstractmethod
    @contextmanager
    def modes(self, inference: bool, cuda: bool) -> 'Pipeline':
        r"""
        Context manager to turn inference and CUDA modes on or off

        :param inference: turn inference mode if True
        :param cuda: copy relevant steps to the GPU if True

        :return: self
        """
        raise NotImplementedError("interface method not implemented")

    @property
    @abstractmethod
    def init_timestamp(self) -> float:
        r"""
        Returns the timestamp created at pipeline initialization

        :return: timestamp as float
        """
        raise NotImplementedError("interface property not implemented")

    @property
    @abstractmethod
    def train_duration_sec(self) -> int:
        r"""
        Returns training duration in seconds

        :return: train seconds
        """
        raise NotImplementedError("interface property not implemented")

    @property
    @abstractmethod
    def is_training(self) -> bool:
        r"""
        Returns True if the pipeline has not finished training

        :return: True if pipeline has not finished training
        """

        raise NotImplementedError("interface property not implemented")

    @staticmethod
    def _import_from_disk(suffix: str, folder: str = "models") -> 'Pipeline':
        with open(f"{folder}/pipeline_{suffix}.pkl", "rb") as f:
            pipeline = pickle.load(f)

        return pipeline

    @staticmethod
    def _import_from_stream(buffer: IO[bytes]) -> 'Pipeline':
        buffer.seek(0)
        return pickle.load(buffer)

    @staticmethod
    def load(suffix:  str | None = None,
             folder:  str = "models",
             buffer: IO[bytes] | None = None) -> PipelineType:
        r"""
        Loads a pipeline from disk or IO buffer.
        Either the suffix+folder or buffer must be set.

        :param suffix: optional string suffix to be used in export
        :param folder: optional directory to be used for exporting
        :param buffer: optional IO buffer used for exporting

        :return: returns self
        """
        if buffer is not None:
            pipeline = Pipeline._import_from_stream(buffer)
        else:
            pipeline = Pipeline._import_from_disk(suffix, folder)

        pipeline.set_inference_mode(on = True, cuda = True)

        return pipeline

    def _export_to_stream(self, buffer: IO[bytes]) -> tuple[str, str, str] | None:
        pickle.dump(self, buffer)

        return None

    def _export_to_disk(self,
                        suffix: str | None = None,
                        folder: str | None = "models") -> tuple[str, str, str] | None:

        suffix = suffix or self.init_timestamp

        filepath_model = f"{folder}/pipeline_{suffix}.pkl"
        filepath_meta  = f"{folder}/metadata_{suffix}.json"

        os.makedirs(folder, exist_ok=True)
        self._logger.info(f"Saving pipeline to disk: '{filepath_model}'")
        self._logger.info(f"Saving metadata to disk: '{filepath_meta}'")

        with open(f"{filepath_model}", "wb") as f:
            pickle.dump(self, f)

        with open(f"{filepath_meta}", "w") as f:
            json.dump(PipelineMetadata.get_metadata(self).to_dict(), f, indent=4)

        return suffix, filepath_meta, filepath_model

    def export(self,
               suffix: str | None = None,
               folder: str | None = "models",
               buffer: IO[bytes] | None = None,
               prune: bool = True) -> tuple[str, str, str] | None:

        r"""Exports a pipeline to disk or IO buffer

        :param suffix: suffix used for exporting
        :param folder: path to the location where the pipeline should be exported
        :param buffer: IO buffer to be used to export to memory instead of disk
        :param prune: should the pipeline be pruned (all transient data deleted) before exporting

        :return: None
        """

        if folder is None and buffer is None:
            raise Exception("Cannot export: folder and file descriptors are missing")

        start_timestamp = datetime.now().timestamp()

        if prune:
            print("Pruning all steps")
            self.clean()

        with self.modes(inference = True, cuda = True):
            if buffer is not None:
                result = self._export_to_stream(buffer)
            else:
                result = self._export_to_disk(suffix, folder)

        self._logger.info(f"Export completed in {datetime.now().timestamp() - start_timestamp:.2f}s.")

        return result

    def deepcopy(self) -> 'Pipeline':
        r"""
        Creates a deep copy of self

        :return: copy of self
        """

        buffer = BytesIO()
        self.export(buffer=buffer, prune=False)
        copy = self._import_from_stream(buffer)
        buffer.close()

        return copy

    def get_signature(self, step_type: Type[PipelineStep], as_string: bool = False, **kwargs) -> dict[str, set[str]] | str:
        r"""
        Builds a signature of a step type along the execution path

        :param step_type: step type (trait, like Trainable, Transformable, etc.)
        :param as_string: return string if True, else dict
        :param kwargs: an optional, not specified list of keyword variables

        :return: signature as a string or a dict of sets of string (metdhod: parameters)
        """

        raise NotImplementedError("interface property not implemented")