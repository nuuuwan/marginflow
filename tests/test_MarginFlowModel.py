import unittest

import numpy as np

from marginflow import MarginFlowModel


class TestMarginFlowModel(unittest.TestCase):
    def setUp(self):
        self.source = np.array(
            [
                [900, 100],
                [100, 900],
                [700, 300],
                [300, 700],
                [500, 500],
            ]
        )
        self.truth = np.array([[0.8, 0.2], [0.25, 0.75]])
        self.target = (self.source @ self.truth).astype(int)

    def test_recovers_transition_matrix_from_dummy_counts(self):
        model = MarginFlowModel().fit(self.source, self.target)

        np.testing.assert_allclose(
            model.transition_matrix_, self.truth, atol=1e-5
        )

    def test_recovers_hand_verifiable_transition_matrix(self):
        source = np.array([[1_000, 0], [0, 1_000], [500, 500]])
        target = np.array([[800, 200], [300, 700], [550, 450]])
        truth = np.array([[0.8, 0.2], [0.3, 0.7]])

        estimate = MarginFlowModel().fit(source, target).transition_matrix_

        np.testing.assert_allclose(estimate, truth, atol=1e-5)

    def test_estimates_are_valid_probabilities_and_flows(self):
        model = MarginFlowModel().fit(self.source, self.target)

        self.assertTrue(np.all(model.transition_matrix_ >= 0))
        np.testing.assert_allclose(model.transition_matrix_.sum(axis=1), 1)
        np.testing.assert_allclose(
            model.predict_counts(self.source), self.target, atol=1e-3
        )
        np.testing.assert_allclose(
            model.aggregate_flows(self.source).sum(axis=1),
            self.source.sum(axis=0),
        )

    def test_rejects_mismatched_group_totals(self):
        target = self.target.copy()
        target[0, 0] += 1

        with self.assertRaisesRegex(ValueError, "totals must match"):
            MarginFlowModel().fit(self.source, target)


if __name__ == "__main__":
    unittest.main()
