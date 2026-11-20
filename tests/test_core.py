import unittest
import numpy as np
from ctf.history import trial_features
from ctf.fusion import equal_family_fusion


class CoreChecks(unittest.TestCase):
    def test_missing_lags_keep_masks(self):
        x = np.zeros((5, 1, 4), dtype=np.float32)
        x[0, 0] = [2, 1, 0, 1]
        x[1:, 0] = [2, 0, 1, 1]
        y = trial_features(x)
        self.assertEqual(y.shape, (5, 1, 400))
        # First lag begins at 4 state features + 156 rolling features.
        self.assertEqual(y[1, 0, 160], 2)
        self.assertEqual(y[1, 0, 161], 1)
        self.assertEqual(y[2, 0, 160], 0)
        self.assertEqual(y[2, 0, 161], 0)

    def test_fusion_is_equal_and_aligned(self):
        np.testing.assert_allclose(equal_family_fusion([1], [2], [3], [4]), [2.5])
        with self.assertRaises(ValueError):
            equal_family_fusion([1], [2, 3], [3], [4])


if __name__ == '__main__':
    unittest.main()
