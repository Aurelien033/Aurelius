"""PowerShell-style object pipeline with chainable transforms.

Inspired by ReactiveX / LINQ.  Fluent pipeline API for processing
collections via filter → map → sort → head → tail → dedup → group_by.

Zero third-party imports — ``collections.abc``, ``typing`` only.

Pattern:
    ``Pipeline([1, 2, 3, 4, 5]).filter(pred).map(f).collect()``
    ``pipeline(source).filter(...).head(5).collect()``
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")
K = TypeVar("K")
U = TypeVar("U")


class Pipeline(Generic[T]):  # noqa: UP046 - keep parseable on Python <3.12
    """Fluent pipeline for chainable collection transforms.

    Each method (filter, map, sort, head, tail, dedup, group_by) returns a
    NEW Pipeline — the original is never mutated.
    """

    def __init__(self, source: list[T]) -> None:
        self._source = source

    def filter(self, predicate: Callable[[T], bool]) -> Pipeline[T]:
        """Keep items where predicate is True.

        >>> Pipeline([1, 2, 3, 4]).filter(lambda x: x > 2).collect()
        [3, 4]
        """
        return Pipeline([item for item in self._source if predicate(item)])

    def map(self, transform: Callable[[T], U]) -> Pipeline[U]:
        """Apply *transform* to every item.

        >>> Pipeline([1, 2, 3]).map(lambda x: x * 2).collect()
        [2, 4, 6]
        """
        return Pipeline([transform(item) for item in self._source])

    def sort(
        self,
        key: Callable[[T], Any] | None = None,
        reverse: bool = False,
    ) -> Pipeline[T]:
        """Sort by *key* (or natural order if None).

        >>> Pipeline([3, 1, 4, 1, 5]).sort().collect()
        [1, 1, 3, 4, 5]
        """
        return Pipeline(sorted(self._source, key=key, reverse=reverse))

    def head(self, count: int) -> Pipeline[T]:
        """Take the first *count* items.

        >>> Pipeline([1, 2, 3, 4, 5]).head(3).collect()
        [1, 2, 3]
        """
        if count <= 0:
            return Pipeline([])
        return Pipeline(self._source[:count])

    def tail(self, count: int) -> Pipeline[T]:
        """Take the last *count* items.

        >>> Pipeline([1, 2, 3, 4, 5]).tail(3).collect()
        [3, 4, 5]
        """
        if count <= 0:
            return Pipeline([])
        return Pipeline(self._source[-count:])

    def dedup(self) -> Pipeline[T]:
        """Remove **consecutive** duplicates only.
        Use ``.sort().dedup()`` to remove all duplicates.

        >>> Pipeline([1, 1, 2, 2, 3, 1]).dedup().collect()
        [1, 2, 3, 1]
        """
        if not self._source:
            return Pipeline([])
        result = [self._source[0]]
        for item in self._source[1:]:
            if item != result[-1]:
                result.append(item)
        return Pipeline(result)

    def group_by(self, key: Callable[[T], K]) -> dict[K, list[T]]:
        """Group items by *key*, returning a ``dict``.

        >>> Pipeline([1, 2, 3, 4, 5, 6]).group_by(lambda x: x % 2)
        {1: [1, 3, 5], 0: [2, 4, 6]}
        """
        groups: dict[K, list[T]] = {}
        for item in self._source:
            k = key(item)
            if k not in groups:
                groups[k] = []
            groups[k].append(item)
        return groups

    def collect(self) -> list[T]:
        """Execute the pipeline and return the result list."""
        return list(self._source)

    def to_pipeline(self) -> Pipeline[T]:
        """Start a new pipeline from current items."""
        return Pipeline(self._source)

    def __iter__(self) -> Iterator[T]:
        return iter(self._source)

    def __len__(self) -> int:
        return len(self._source)

    def __repr__(self) -> str:
        return f"Pipeline({self._source!r})"


def pipeline(source: list[T]) -> Pipeline[T]:  # noqa: UP047 - Python <3.12 compat
    """Factory: ``Pipeline(source)`` is equivalent to ``pipeline(source)``."""
    return Pipeline(source)
