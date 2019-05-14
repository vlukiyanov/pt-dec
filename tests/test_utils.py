import numpy as np
import torch
from unittest import TestCase

from ptdec.utils import cluster_accuracy, target_distribution


class TestClusterAccuracy(TestCase):
    def test_basic(self):
        """
        Basic test to check that the calculation is sensible.
        """
        true_value1 = np.array([1, 2, 1, 2, 0, 0], dtype=np.int64)
        pred_value1 = np.array([2, 1, 2, 1, 0, 0], dtype=np.int64)
        self.assertAlmostEqual(
            cluster_accuracy(true_value1, pred_value1)[1],
            1.0
        )
        self.assertAlmostEqual(
            cluster_accuracy(true_value1, pred_value1, 3)[1],
            1.0
        )
        self.assertDictEqual(
            cluster_accuracy(true_value1, pred_value1)[0],
            {0: 0, 1: 2, 2: 1}
        )
        true_value2 = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
        pred_value2 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        self.assertAlmostEqual(
            cluster_accuracy(true_value2, pred_value2)[1],
            1.0 / 6.0
        )
        self.assertAlmostEqual(
            cluster_accuracy(true_value2, pred_value2, 6)[1],
            1.0 / 6.0
        )
        true_value3 = np.array([1, 3, 1, 3, 0, 2], dtype=np.int64)
        pred_value3 = np.array([2, 1, 2, 1, 3, 0], dtype=np.int64)
        self.assertDictEqual(
            cluster_accuracy(true_value3, pred_value3)[0],
            {2: 1, 1: 3, 3: 0, 0: 2}
        )


class TestTargetDistribution(TestCase):
    def test_basic(self):
        """
        Basic test to check that the calculation is sensible and conforms to the formula.
        """
        test_tensor = torch.Tensor(
            [
                [0.5, 0.5],
                [0.0, 1.0]
            ]
        )
        output = target_distribution(test_tensor)
        self.assertAlmostEqual(
            tuple(output[0]),
            (0.75, 0.25)
        )
        self.assertAlmostEqual(
            tuple(output[1]),
            (0.0, 1.0)
        )
