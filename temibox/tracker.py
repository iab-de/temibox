import logging
from typing import Callable


class Tracker:
    r"""
    Generic activity tracker

    Can be used by pipeline steps to track activities.
    """

    logger = logging.getLogger("Tracker")

    def __init__(self):
        self._task = "<noname>"
        self._progress_fns = []
        self._log_fns = []

    def task(self, task: str) -> None:
        r"""
        Logs the begining of a task

        :param task: title of the task

        :return: None
        """

        self._task = task
        self.log(f"Starting task '{task}'")

    def progress(self,
                 i: int,
                 total: int,
                 message: str = "") -> None:
        r"""
        Logs progress and notifies all provided progress trackers

        :param i: current step (if makes sense)
        :param total: total number of steps (if known and makes sense)
        :param message: progress message

        :return: None
        """

        message_text = f" ({message})" if len(message.strip()) else ""
        ratio = f"{i/total:.2%}" if total > 0 else ""
        self.log(f"Progress on task '{self._task}': {ratio}{message_text}")

        for progress_fn in self._progress_fns:
            progress_fn(i, total, message)

    def add_progress_callback(self, progress_fn: Callable[[int, int, str], None]) -> None:
        r"""
        Adds a progress callback to the list of callbacks

        This method can be used to add additional progress notification methods
        (this is similar to the observer pattern)

        :param progress_fn: a progress method with the same signature as self.progress

        :return: None
        """

        self._progress_fns.append(progress_fn)

    def clear_progress_callbacks(self) -> None:
        r"""
        Clears the progress callback list

        :return: None
        """

        self._progress_fns = []

    def add_log_callback(self, log_fn: Callable[[str], None]) -> None:
        r"""
        Adds a logging callback to the list of callback

        :param log_fn: a logging method with the same signature as self.log

        :return: None
        """

        self._log_fns.append(log_fn)

    def clear_log_callbacks(self) -> None:
        r"""
        Clears the logging callback list

        :return: None
        """
        self._log_fns = []

    def log(self, message: str) -> None:
        r"""
        Logs a message

        :param message: a message

        :return: None
        """

        self.logger.info(message)

        for fn in self._log_fns:
            fn(message)
