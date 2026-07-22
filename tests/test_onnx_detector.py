import unittest

import numpy as np

from src.tracking.onnx_detector import OnnxBallDetector


class OnnxDecoderTest(unittest.TestCase):
    def setUp(self):
        self.detector = OnnxBallDetector("unused.onnx")

    def test_decodes_end_to_end_output(self):
        output = np.array([[[1, 2, 10, 12, 0.9, 0], [3, 4, 8, 9, 0.2, 0]]])
        boxes, scores, classes, end_to_end = self.detector._decode(output)
        self.assertTrue(end_to_end)
        self.assertEqual(boxes.shape, (2, 4))
        self.assertAlmostEqual(scores[0], 0.9)
        self.assertEqual(classes.tolist(), [0, 0])

    def test_decodes_raw_one_class_output(self):
        output = np.zeros((1, 5, 10), dtype=np.float32)
        output[0, :4, 0] = [20, 30, 8, 10]
        output[0, 4, 0] = 0.8
        boxes, scores, classes, end_to_end = self.detector._decode(output)
        self.assertFalse(end_to_end)
        self.assertEqual(boxes.shape, (10, 4))
        self.assertAlmostEqual(float(scores[0]), 0.8, places=5)
        self.assertEqual(classes[0], 0)


if __name__ == "__main__":
    unittest.main()
