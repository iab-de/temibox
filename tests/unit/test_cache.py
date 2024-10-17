import pytest

from temibox.cache import Cache


def test_properties_cache():
    cache = Cache(max_entries = 3)

    assert cache.size == 0, "Empty cache should have size 0"
    assert cache.max_size == 3, "Cache size should be equal to 3"
    assert cache.is_caching, "Cache is naturally always caching"
    assert cache.cache == cache, "Cache.cache returns self"

def test_simple_inserts():
    cache = Cache(max_entries = 3)

    cache.add("test1", 1)
    cache.add("test2", 2)
    cache.add("test3", 3)

    assert cache.size == 3, "Cache should have 3 items"

def test_repeated_inserts():
    cache = Cache(max_entries = 3)

    cache.add("test1", 1)
    cache.add("test1", 2)
    cache.add("test1", 3)

    assert cache.size == 1, "Cache should have 1 items"
    assert cache.get("test1") == 3, "Repeated inserts should change the value"

def test_pruning():
    cache = Cache(max_entries = 2)

    cache.add("test1", 1)
    cache.add("test2", 2)
    cache.add("test3", 3)
    cache.add("test4", 4)

    assert cache.max_size == 2, "Cache should have size 2"
    assert cache.size == 2, "Cache should have 2 items"
    assert cache.get("test1") is None and cache.get("test2") is None, "First two entries should've been evicted"
    assert cache.get("test3") == 3, "Invalid value (expected 3)"
    assert cache.get("test4") == 4, "Invalid value (expected 3)"

def test_order():
    cache = Cache(max_entries = 10)

    for i in range(10, 1, -1):
        cache.add(f"test{i}", i)
        cache.add(f"test{i}", i)

    entry = cache._lru_first_entry
    for i in range(0, 9):
        assert entry.value == 10-i, f"Invalid value {i}"
        entry = entry.next_entry

    assert entry is None, "Last entry should not have a next entry"

def test_add_to_end():
    cache = Cache(max_entries = 10)

    for i in range(1, 6):
        cache.add(f"test{i}", i)
        cache.add(f"test{i}", i)

    assert cache._lru_first_entry.value == 1, "Invalid value for the first entry"
    assert cache._lru_last_entry.value == 5, "Invalid value for the second entry"

    cache.add("test6", 99)
    assert cache._lru_last_entry.value == 99, "Invalid value for the newly inserted entry"
    assert cache._lru_last_entry.prev_entry.value == 5, "Previous entry invalid"
    assert cache._lru_first_entry.value == 1, "First value remains"

def _assert_order(cache, expected):
    entry = cache._lru_first_entry
    for x in expected:
        assert entry.value == x, "Order does not match"
        entry = entry.next_entry

    assert entry is None, "Last entry should be None"

def test_updating():

    cache = Cache(max_entries=10)
    cache.add("test1", 1)
    cache.add("test2", 2)
    cache.add("test3", 3)
    cache.add("test4", 4)

    _assert_order(cache, [1, 2, 3, 4])

    cache.get("test1")
    _assert_order(cache, [2, 3, 4, 1])
    cache.get("test1")
    _assert_order(cache, [2, 3, 4, 1])

    cache.get("test2")
    _assert_order(cache, [3, 4, 1, 2])

    cache.add("test4", 4)
    _assert_order(cache, [3, 1, 2, 4])

    cache.add("test5", 5)
    _assert_order(cache, [3, 1, 2, 4, 5])

    cache.get("test3")
    cache.get("test4")
    cache.get("test5")
    _assert_order(cache, [1, 2, 3, 4, 5])

def test_clearing():
    cache = Cache(max_entries = 2)

    cache.add("test1", 1)
    cache.add("test2", 2)

    assert cache.size == 2, "Cache size should be 2"
    cache.clear_cache()
    assert cache.size == 0, "Cleared cache size should be 0"

def test_changing_cache_size():
    cache = Cache(max_entries = 2)

    cache.add("test1", 1)
    cache.add("test2", 2)

    assert cache.max_size == 2, "Cache max size should be 2"
    assert cache.size == 2, "Cache size should be 2"
    assert cache._lru_first_entry.value == 1, "Invalid first entry"
    assert cache._lru_last_entry.value == 2, "Invalid second entry"

    cache.configure_cache(on = True, max_entries=10)
    assert cache.max_size == 10, "Cache max size should be 10"
    assert cache.size == 2, "Cache size should be 2"
    assert cache._lru_first_entry.value == 1, "Invalid first entry"
    assert cache._lru_last_entry.value == 2, "Invalid second entry"

    cache.clear_cache()
    assert cache.max_size == 10, "Cache max size should be 10"
    assert cache.size == 0, "Cleared cache size should be 0"

def test_errors():

    with pytest.raises(Exception):
        Cache(max_entries=0)

    with pytest.raises(Exception):
        Cache().configure_cache(on=True, max_entries = -1)

def test_pickling():
    import pickle

    class IO:
        def __init__(self):
            self._data = None

        def read(self, *args):
            return self._data

        def readline(self):
            return self._data

        def write(self, data):
            self._data = data

    io = IO()
    c1 = Cache(max_entries=4)
    c1.add("test", 42)

    pickle.dump(c1, io)

    c2 = pickle.load(io)

    assert c1 != c2, "Should not be the same object"
    assert c1.get("test") == c2.get("test") == 42, "Recovered value is wrong"

def test_prune():
    cache = Cache(max_entries=2)

    cache.add("test1", 1)
    cache.add("test2", 2)
    cache.add("test3", 3)

    assert cache.get("test1") is None, "First entry should have been removed"
    assert cache.get("test2") == 2, "Second entry is wrong"
    assert cache.get("test3") == 3, "Third entry is wrong"

    cache.configure_cache(on=False)
