import numpy as np
from scipy.optimize import minimize

from marginflow.MarginFlowModel.MarginFlowMathMixin import MarginFlowMathMixin
from marginflow.MarginFlowModel.MarginFlowValidationMixin import (
    MarginFlowValidationMixin,
)


class MarginFlowModel(MarginFlowMathMixin, MarginFlowValidationMixin):
    """Estimate how people likely moved between categories.

    The model learns movement probabilities from group totals measured before
    and after a change. It never needs records linking individual people.

    Attributes:
        transition_matrix_: Learned movement probabilities after fitting.
    """

    def __init__(self, max_iterations=1_000, tolerance=1e-10):
        """Configure how thoroughly the model searches for an answer.

        Args:
            max_iterations: Maximum number of improvements to attempt.
            tolerance: Smallest improvement worth continuing the search for.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.transition_matrix_ = None

    def fit(self, source_counts, target_counts):
        """Learn movement rates that best reproduce the final totals.

        Args:
            source_counts: Starting category counts arranged by group.
            target_counts: Final category counts arranged by group.

        Returns:
            This fitted model.

        Raises:
            ValueError: If the two snapshots are not comparable.
            RuntimeError: If the numerical search cannot find an answer.
        """
        source, target = self._validated_counts(source_counts, target_counts)
        source_shares = source / source.sum(axis=1, keepdims=True)
        shape = (source.shape[1], target.shape[1] - 1)
        initial = np.zeros(np.prod(shape), dtype=float)
        result = minimize(
            self._objective,
            initial,
            args=(shape, source_shares, target),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )
        if not result.success:
            raise RuntimeError(f"Transition fitting failed: {result.message}")
        self.transition_matrix_ = self._transition(result.x, shape)
        return self

    def predict_proportions(self, source_counts):
        """Estimate each group's final category shares.

        Args:
            source_counts: Starting category counts arranged by group.

        Returns:
            Predicted final proportions for every group.
        """
        source = self._validated_source(source_counts)
        self._require_fitted(source.shape[1])
        shares = source / source.sum(axis=1, keepdims=True)
        return shares @ self.transition_matrix_

    def predict_counts(self, source_counts):
        """Estimate each group's final number of people per category.

        Args:
            source_counts: Starting category counts arranged by group.

        Returns:
            Predicted final counts for every group.
        """
        source = self._validated_source(source_counts)
        proportions = self.predict_proportions(source)
        return proportions * source.sum(axis=1, keepdims=True)

    def aggregate_flows(self, source_counts):
        """Estimate total movement between every pair of categories.

        Args:
            source_counts: Starting category counts arranged by group.

        Returns:
            Expected people moving from each source to each target category.
        """
        source = self._validated_source(source_counts)
        self._require_fitted(source.shape[1])
        return source.sum(axis=0)[:, None] * self.transition_matrix_
