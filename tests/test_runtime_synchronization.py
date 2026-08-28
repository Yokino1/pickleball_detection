import unittest

from src.runtime import (
    BoundedLatestQueue,
    FramePacket,
    FramePair,
    TimestampPairer,
)


def packet(
    side: str,
    sequence: int,
    timestamp_s: float,
) -> FramePacket:
    return FramePacket(
        side=side,
        sequence=sequence,
        capture_timestamp_s=timestamp_s,
        frame=f"{side}-{sequence}",
    )


class RuntimeSynchronizationTest(unittest.TestCase):
    def test_frame_packet_validates_side_sequence_and_timestamp(self):
        with self.assertRaises(ValueError):
            packet("center", 0, 0.0)
        with self.assertRaises(ValueError):
            packet("left", -1, 0.0)
        with self.assertRaises(ValueError):
            packet("right", 0, float("nan"))

    def test_frame_pair_reports_average_timestamp_and_signed_skew(self):
        pair = FramePair(
            packet("left", 0, 1.000),
            packet("right", 0, 1.003),
        )

        self.assertAlmostEqual(pair.timestamp_s, 1.0015)
        self.assertAlmostEqual(pair.skew_ms, 3.0)

    def test_bounded_queue_drops_oldest_at_capacity(self):
        queue = BoundedLatestQueue[int](capacity=2)
        queue.put(1)
        queue.put(2)
        dropped = queue.put(3)

        self.assertEqual(dropped, 1)
        self.assertEqual(queue.pop_oldest(), 2)
        self.assertEqual(queue.diagnostics()["capacity_drops"], 1)

    def test_pop_latest_discards_superseded_pending_items(self):
        queue = BoundedLatestQueue[int](capacity=4)
        for value in (1, 2, 3):
            queue.put(value)

        self.assertEqual(queue.pop_latest(), 3)
        self.assertEqual(len(queue), 0)
        self.assertEqual(queue.diagnostics()["superseded_drops"], 2)

    def test_pairs_frames_inside_capture_skew_tolerance(self):
        pairer = TimestampPairer(max_skew_ms=5.0)

        self.assertIsNone(pairer.push(packet("left", 0, 1.000)))
        pair = pairer.push(packet("right", 0, 1.004))

        self.assertIsNotNone(pair)
        self.assertEqual(pair.left.sequence, 0)
        self.assertEqual(pair.right.sequence, 0)
        self.assertAlmostEqual(pair.skew_ms, 4.0)
        self.assertEqual(pairer.diagnostics()["paired_count"], 1)

    def test_discards_older_frame_until_a_pair_is_possible(self):
        pairer = TimestampPairer(max_skew_ms=3.0)
        pairer.push(packet("left", 0, 1.000))
        pairer.push(packet("left", 1, 1.016))

        pair = pairer.push(packet("right", 0, 1.018))

        self.assertIsNotNone(pair)
        self.assertEqual(pair.left.sequence, 1)
        self.assertEqual(pair.right.sequence, 0)
        self.assertEqual(
            pairer.diagnostics()["stale_drops"]["left"],
            1,
        )

    def test_rejects_duplicate_or_out_of_order_packets(self):
        pairer = TimestampPairer()
        pairer.push(packet("left", 2, 1.0))

        self.assertIsNone(pairer.push(packet("left", 1, 1.1)))
        self.assertIsNone(pairer.push(packet("left", 3, 0.9)))
        self.assertEqual(
            pairer.diagnostics()["out_of_order_drops"]["left"],
            2,
        )

    def test_pairer_queue_capacity_prefers_recent_frames(self):
        pairer = TimestampPairer(max_skew_ms=2.0, queue_capacity=2)
        pairer.push(packet("left", 0, 0.000))
        pairer.push(packet("left", 1, 0.016))
        pairer.push(packet("left", 2, 0.032))

        pair = pairer.push(packet("right", 0, 0.033))

        self.assertIsNotNone(pair)
        self.assertEqual(pair.left.sequence, 2)
        self.assertEqual(
            pairer.diagnostics()["capacity_drops"]["left"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
