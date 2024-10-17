from dataclasses import dataclass

from .domain import UseCase

from typing import TypeVar, Generic, Optional, Union

T = TypeVar("T")

@dataclass
class Context:
    r"""
    Pipeline execution context

    Contains a reference to the pipeline, all of its usecase,
    the active usecase and the name of the active step
    """

    pipeline: 'Pipeline'
    usecases: list[UseCase]
    active_usecase: Optional[UseCase] = None
    active_step_name: Optional[str] = None

class ContextArg(Generic[T]):
    r"""
    Wrapper type to contain the same type of value in different contexts (usecases)
    """

    def __init__(self, usecase_name: str = None, arg: T = None):
        self._args: dict[str, T] = {}
        if usecase_name and arg:
            self.add(usecase_name, arg)

    @staticmethod
    def extract(obj: Union['ContextArg[T]', T], usecase_name: str = None) -> Optional[T]:
        r"""
        Extracts the value of type T from the wrapper.

        ContextArg.extract is safe to use on non-wrapped objects

        If no usecase name is provided, then the first available value will be extracted.

        :param obj: wrapped or non-wrapped object
        :param usecase_name: optional usecase name

        :return: value of type T or None
        """
        if isinstance(obj, ContextArg):
            if usecase_name:
                return obj.get(usecase_name)
            else:
                return obj.values[0] if obj.values and len(obj.values) else None

        return obj

    def add(self, usecase_name: str, payload: T) -> 'ContextArg':
        r"""
        Add a value of type T to the wrapper

        If a value for the provided usecase name already exists,
        the value will be overwritten

        :param usecase_name: name of the usecase
        :param payload: value of tpye T

        :return: self (for method chaining)
        """

        self._args[usecase_name.lower().strip()] = payload
        return self

    def get(self, usecase_name: str) -> Optional[T]:
        r"""
        Returns an unwrapped value of type T for a given usecase

        :param usecase_name: name of the usecase

        :return: value of type T or None
        """
        return self._args.get(usecase_name.lower().strip(), None)

    @property
    def values(self) -> list[T]:
        r"""
        Returns all the unwrapped values

        :return: list of values of type T
        """
        return list(self._args.values())
