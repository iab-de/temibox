import hashlib
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar, Generic, Optional, Any

from ..traits import Cacheable

T = TypeVar("T")

@dataclass
class LRUEntry:
    cache_key: str
    value: T
    timestamp: int

    prev_entry: Optional['LRUEntry'] = None
    next_entry: Optional['LRUEntry'] = None

class Cache(Cacheable, Generic[T]):
    r"""
    Simple fixed size LRU cache
    """

    def __init__(self, max_entries: int):
        super().__init__()

        self._cache_mutex = threading.Lock()
        self._max_entries = max_entries
        self._cache: dict[str, LRUEntry] = {}
        self._lru_first_entry: Optional[LRUEntry] = None
        self._lru_last_entry:  Optional[LRUEntry] = None

        if max_entries < 1:
            raise Exception("Invalid value for max_entries")

    def _prune_cache(self):
        assert self._cache_mutex.locked(), "Invalid use of method: cache mutex should be locked"

        while len(self._cache) >= self._max_entries:
            entry = self._lru_first_entry

            if entry is None:
                break

            if nentry := entry.next_entry:
                nentry.prev_entry = None
                self._lru_first_entry = nentry

            del self._cache[entry.cache_key]

        if not len(self._cache):
            self._lru_first_entry = None
            self._lru_last_entry  = None

    def _get_cache_key(self, needle: Any) -> str:
        key = f"{type(needle).__name__}::'{str(needle)}'"
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    def _update(self, entry: LRUEntry, value: T):
        assert self._cache_mutex.locked(), "Invalid use of method: cache mutex should be locked"

        entry.timestamp = int(datetime.now().timestamp())
        entry.value     = value
        pentry = entry.prev_entry
        nentry = entry.next_entry

        # Connect previous entry with next entry
        if nentry:
            if pentry:
                pentry.next_entry = nentry
            else:
                self._lru_first_entry = nentry

            nentry.prev_entry = pentry

        # Push entry to the back if necessary
        if (lentry := self._lru_last_entry).cache_key != entry.cache_key:
            lentry.next_entry = entry
            entry.prev_entry = lentry
            entry.next_entry = None
            self._lru_last_entry = entry

    def __setstate__(self, state):
        self.__dict__ = state
        self._cache_mutex = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()

        del state["_cache_mutex"]
        return state

    @property
    def size(self) -> int:
        r"""
        Returns the number of elements in the cache

        :return: number of elements in the cache
        """
        with self._cache_mutex:
            return len(self._cache)

    @property
    def max_size(self) -> int:
        r"""
        Returns the capacity of the cache

        :return: capacity of the cache
        """
        with self._cache_mutex:
            return self._max_entries

    def configure_cache(self, on: bool, max_entries: int = 1024):
        if not on:
            self.clear_cache()
        else:
            if max_entries <= 0:
                raise Exception("Invalid max_entries value. Expecting >= 1")

            with self._cache_mutex:
                self._max_entries = max_entries
                self._prune_cache()

    def add(self, needle: Any, value: T) -> None:
        r"""
        Ads an item to the cache

        :param needle: key used to identify value (can be equal to value, needs to be hashable)
        :param value: value to be cached

        :return: None
        """

        with self._cache_mutex:
            cache_key = self._get_cache_key(needle)

            if cache_key in self._cache:
                self._update(self._cache[cache_key], value)
                return

            if len(self._cache) >= self._max_entries:
                self._prune_cache()

            entry = LRUEntry(cache_key = cache_key,
                             value     = value,
                             timestamp = int(datetime.now().timestamp()))

            if lentry := self._lru_last_entry:
                entry.prev_entry = lentry
                lentry.next_entry = entry
                self._lru_last_entry = entry
            elif not self._lru_first_entry:
                self._lru_first_entry = entry
                self._lru_last_entry  = entry

            self._cache[cache_key] = entry

    def get(self, needle: Any) -> Optional[T]:
        r"""
        Return an element from the cache if it exists

        :param needle: key

        :return: element of type T if it exists
        """

        value = None
        with self._cache_mutex:
            entry = self._cache.get(self._get_cache_key(needle), None)
            if entry is not None:
                value = entry.value
                self._update(entry, value)

        return value

    def clear_cache(self):
        with self._cache_mutex:
            self._cache: dict[str, LRUEntry] = {}
            self._lru_first_entry = None
            self._lru_last_entry = None

    @property
    def cache(self) -> 'Cache':
        return self

    @property
    def is_caching(self) -> bool:
        return True