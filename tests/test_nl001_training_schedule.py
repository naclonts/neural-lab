import unittest

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed')
class TrainingScheduleTests(unittest.TestCase):
    def test_deterministic_epochs_and_partial_batch(self):
        from neural_lab.nl001_training import batch_indices
        schedule = list(batch_indices(7, 3, 6, 42))
        self.assertEqual(schedule, list(batch_indices(7, 3, 6, 42)))
        self.assertEqual([len(x) for x in schedule], [3, 3, 1, 3, 3, 1])
        self.assertEqual(sorted(sum(schedule[:3], [])), list(range(7)))
        self.assertEqual(sorted(sum(schedule[3:], [])), list(range(7)))
        with self.assertRaises(ValueError):
            list(batch_indices(7, 0, 1, 42))

    def test_full_training_selection_weights_partial_batch_and_rejects_nonfinite(self):
        from unittest.mock import patch
        import torch
        from neural_lab.nl001_training import full_training_cross_entropy

        model = torch.nn.Linear(1, 1)
        model.train()
        records = [1.0, 2.0, 9.0]
        def batch(items, indices, device):
            return torch.tensor([items[i] for i in indices]), None
        def loss(current, inputs, targets, *, with_cross_entropy):
            self.assertFalse(current.training)
            self.assertFalse(torch.is_grad_enabled())
            value = inputs.mean()
            return value, value
        with patch('neural_lab.nl001_training._batch', side_effect=batch), patch('neural_lab.nl001_training.training_loss', side_effect=loss):
            self.assertEqual(full_training_cross_entropy(model, records, 2, 'cpu'), 4.0)
            with self.assertRaises(ValueError):
                full_training_cross_entropy(model, [float('nan')], 2, 'cpu')
