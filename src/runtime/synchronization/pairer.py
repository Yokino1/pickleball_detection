"""Bounded timestamp pairing for concurrent left/right capture workers."""

from __future__ import annotations

from threading import RLock
from typing import Optional

from ..frame_packet import FramePacket, FramePair
from .queues import BoundedLatestQueue


class TimestampPairer:
    """Pair left/right frames by capture timestamp and discard stale inputs."""

    def __init__(
        self,
        max_skew_ms: float = 5.0,
        queue_capacity: int = 4,
    ):
        if float(max_skew_ms) < 0.0:
            raise ValueError("TimestampPairer.max_skew_ms must be non-negative")
        self.max_skew_s = float(max_skew_ms) / 1000.0
        self.left_queue: BoundedLatestQueue[FramePacket] = (
            BoundedLatestQueue(queue_capacity)
        )
        self.right_queue: BoundedLatestQueue[FramePacket] = (
            BoundedLatestQueue(queue_capacity)
        )
        self._lock = RLock()
        self._last_sequence = {"left": -1, "right": -1}
        self._last_timestamp_s = {
            "left": float("-inf"),
            "right": float("-inf"),
        }
        self.paired_count = 0
        self.stale_drops = {"left": 0, "right": 0}
        self.out_of_order_drops = {"left": 0, "right": 0}
        self.capacity_drops = {"left": 0, "right": 0}
        self.last_skew_ms: Optional[float] = None
        self.max_abs_skew_ms = 0.0

    def push(self, packet: FramePacket) -> Optional[FramePair]:
        """Accept one captured frame and return a pair when one is available."""
        with self._lock:
            side = packet.side
            if (
                packet.sequence <= self._last_sequence[side]
                or packet.capture_timestamp_s
                <= self._last_timestamp_s[side]
            ):
                self.out_of_order_drops[side] += 1
                return None
            self._last_sequence[side] = int(packet.sequence)
            self._last_timestamp_s[side] = float(
                packet.capture_timestamp_s
            )
            queue = (
                self.left_queue if side == "left" else self.right_queue
            )
            if queue.put(packet) is not None:
                self.capacity_drops[side] += 1
            return self._try_pair()

    def reset(self) -> None:
        with self._lock:
            self.left_queue.clear()
            self.right_queue.clear()
            self._last_sequence = {"left": -1, "right": -1}
            self._last_timestamp_s = {
                "left": float("-inf"),
                "right": float("-inf"),
            }
            self.paired_count = 0
            self.stale_drops = {"left": 0, "right": 0}
            self.out_of_order_drops = {"left": 0, "right": 0}
            self.capacity_drops = {"left": 0, "right": 0}
            self.last_skew_ms = None
            self.max_abs_skew_ms = 0.0

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "max_skew_ms": self.max_skew_s * 1000.0,
                "paired_count": self.paired_count,
                "last_skew_ms": self.last_skew_ms,
                "max_abs_skew_ms": self.max_abs_skew_ms,
                "stale_drops": dict(self.stale_drops),
                "out_of_order_drops": dict(
                    self.out_of_order_drops
                ),
                "capacity_drops": dict(self.capacity_drops),
                "left_queue": self.left_queue.diagnostics(),
                "right_queue": self.right_queue.diagnostics(),
            }

    def _try_pair(self) -> Optional[FramePair]:
        while len(self.left_queue) and len(self.right_queue):
            left = self.left_queue.peek_oldest()
            right = self.right_queue.peek_oldest()
            if left is None or right is None:
                return None
            skew_s = (
                float(right.capture_timestamp_s)
                - float(left.capture_timestamp_s)
            )
            if abs(skew_s) <= self.max_skew_s:
                paired_left = self.left_queue.pop_oldest()
                paired_right = self.right_queue.pop_oldest()
                if paired_left is None or paired_right is None:
                    return None
                pair = FramePair(paired_left, paired_right)
                self.paired_count += 1
                self.last_skew_ms = pair.skew_ms
                self.max_abs_skew_ms = max(
                    self.max_abs_skew_ms,
                    abs(pair.skew_ms),
                )
                return pair
            if skew_s > 0.0:
                self.left_queue.pop_oldest()
                self.stale_drops["left"] += 1
            else:
                self.right_queue.pop_oldest()
                self.stale_drops["right"] += 1
        return None
