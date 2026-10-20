import unittest

import torch

from src.mfmp import calculate_mfmp_from_hidden, calculate_sentence_statistics


class MFMPTests(unittest.TestCase):
    def test_population_z_scores_exclude_padding(self):
        hidden = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])
        mean, magnitudes, abs_z, counts = calculate_sentence_statistics(hidden, mask)
        torch.testing.assert_close(mean, torch.tensor([[0.5, 1.0]]))
        torch.testing.assert_close(magnitudes[:, :2], torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(abs_z[:, :2], torch.tensor([[1.0, 1.0]]))
        self.assertEqual(counts.item(), 2.0)

    def test_inclusive_bounds_pool_original_vectors(self):
        hidden = torch.tensor([[[1.0, 0.0], [0.0, 3.0], [4.0, 4.0]]])
        mask = torch.tensor([[1, 1, 0]])
        mean, _, abs_z, _ = calculate_sentence_statistics(hidden, mask)
        pooled, retained, fallback = calculate_mfmp_from_hidden(
            hidden, mask, mean, abs_z, 1.0, 1.0
        )
        torch.testing.assert_close(pooled, mean)
        self.assertEqual(retained.item(), 2)
        self.assertFalse(fallback.item())

    def test_empty_mask_falls_back_to_unfiltered_mean(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [8.0, 8.0]]])
        mask = torch.tensor([[1, 1, 0]])
        mean, _, abs_z, _ = calculate_sentence_statistics(hidden, mask)
        pooled, retained, fallback = calculate_mfmp_from_hidden(
            hidden, mask, mean, abs_z, 2.0, 3.0
        )
        torch.testing.assert_close(pooled, mean)
        self.assertEqual(retained.item(), 0)
        self.assertTrue(fallback.item())

    def test_invalid_threshold_order_is_rejected(self):
        hidden = torch.zeros(1, 1, 2)
        mask = torch.ones(1, 1, dtype=torch.long)
        mean, _, abs_z, _ = calculate_sentence_statistics(hidden, mask)
        with self.assertRaises(ValueError):
            calculate_mfmp_from_hidden(hidden, mask, mean, abs_z, 2.0, 1.0)


if __name__ == "__main__":
    unittest.main()
