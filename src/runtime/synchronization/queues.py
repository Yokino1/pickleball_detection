"""Thread-safe bounded queues that discard stale frames before they accumulate."""

from __future__ import annotations

from collections import deque
from threading import RLock
from typing import Generic, Optional, TypeVar


ItemT = TypeVar("ItemT")


class BoundedLatestQueue(Generic[ItemT]):
    """Keep a bounded FIFO while offering an explicit latest-item pop.

    Capacity overflow always removes the oldest item. ``pop_latest`` removes
    any older pending items as superseded work, which is useful when inference
    falls behind live capture.
    """

    def __init__(self, capacity: int):
        if int(capacity) < 1:
            raise ValueError("BoundedLatestQueue.capacity must be at least 1")
        self.capacity = int(capacity)
        self._items: deque[ItemT] = deque()
        self._lock = RLock()
        self.capacity_drops = 0
        self.superseded_drops = 0

    def put(self, item: ItemT) -> Optional[ItemT]:
        """Append an item and return the oldest item if capacity dropped it."""
        with self._lock:
            dropped = None
            if len(self._items) >= self.capacity:
                dropped = self._items.popleft()
                self.capacity_drops += 1
            self._items.append(item)
            return dropped

    def peek_oldest(self) -> Optional[ItemT]:
        with self._lock:
            return self._items[0] if self._items else None

    def pop_oldest(self) -> Optional[ItemT]:
        with self._lock:
            return self._items.popleft() if self._items else None

    def pop_latest(self) -> Optional[ItemT]:
        """Return the newest item and count all older pending items as stale."""
        with self._lock:
            if not self._items:
                return None
            latest = self._items.pop()
            self.superseded_drops += len(self._items)
            self._items.clear()
            return latest

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "capacity": self.capacity,
                "pending": len(self._items),
                "capacity_drops": self.capacity_drops,
                "superseded_drops": self.superseded_drops,
            }
