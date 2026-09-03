import numpy as np
from scipy.special import softmax


class MarginFlowMathMixin:
    """Provide the probability calculations used by the estimator."""

    def _objective(self, parameters, shape, source_shares, target):
        """Score one candidate matrix and show how to improve it.

        A lower score means the candidate better explains the observed final
        counts. The gradient points the optimiser toward a lower score.

        Args:
            parameters: Unrestricted numbers representing movement rates.
            shape: Number of source rows and free target columns.
            source_shares: Starting category shares for every group.
            target: Observed final category counts for every group.

        Returns:
            The candidate's score and gradient.
        """
        transition = self._transition(parameters, shape)
        probabilities = np.clip(source_shares @ transition, 1e-12, 1.0)
        value = -np.sum(target * np.log(probabilities))
        probability_gradient = source_shares.T @ (-target / probabilities)
        row_total = np.sum(probability_gradient * transition, axis=1)
        logit_gradient = transition * (
            probability_gradient - row_total[:, None]
        )
        return value, logit_gradient[:, :-1].ravel()

    def _transition(self, parameters, shape):
        """Turn unrestricted numbers into valid movement probabilities.

        Args:
            parameters: Unrestricted numbers chosen by the optimiser.
            shape: Number of source rows and free target columns.

        Returns:
            A matrix containing non-negative rows that each sum to one.
        """
        reduced = parameters.reshape(shape)
        logits = np.column_stack((reduced, np.zeros(shape[0])))
        return softmax(logits, axis=1)
