import numpy as np


class MarginFlowValidationMixin:
    """Check that inputs describe usable and comparable populations."""

    def _validated_counts(self, source_counts, target_counts):
        """Check that both snapshots contain matching groups and totals.

        Args:
            source_counts: Starting category counts arranged by group.
            target_counts: Final category counts arranged by group.

        Returns:
            Numeric source and target matrices ready for fitting.

        Raises:
            ValueError: If groups or population totals do not match.
        """
        source = self._validated_source(source_counts)
        target = self._as_counts(target_counts, "target_counts")
        if source.shape[0] != target.shape[0]:
            raise ValueError("Source and target must contain the same groups")
        if not np.allclose(source.sum(axis=1), target.sum(axis=1)):
            raise ValueError("Source and target totals must match by group")
        return source, target

    def _validated_source(self, source_counts):
        """Check that every source group contains at least one person.

        Args:
            source_counts: Starting category counts arranged by group.

        Returns:
            A numeric source matrix.

        Raises:
            ValueError: If a group is empty or the counts are invalid.
        """
        source = self._as_counts(source_counts, "source_counts")
        if np.any(source.sum(axis=1) == 0):
            raise ValueError("Every group must contain at least one member")
        return source

    def _as_counts(self, values, name):
        """Convert count-like input into a checked numeric matrix.

        Args:
            values: Values to interpret as category counts.
            name: Input name used in error messages.

        Returns:
            A floating-point matrix containing the counts.

        Raises:
            ValueError: If values are missing, negative, or not whole numbers.
        """
        counts = np.asarray(values)
        if counts.ndim != 2 or 0 in counts.shape:
            raise ValueError(f"{name} must be a non-empty matrix")
        if not np.issubdtype(counts.dtype, np.number):
            raise ValueError(f"{name} must contain numeric counts")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError(f"{name} must contain valid counts")
        if not np.all(counts == np.floor(counts)):
            raise ValueError(f"{name} must contain integer counts")
        return counts.astype(float)

    def _require_fitted(self, source_categories):
        """Ensure predictions use a fitted model with matching categories.

        Args:
            source_categories: Number of source categories in new data.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If the new data has a different category count.
        """
        if self.transition_matrix_ is None:
            raise RuntimeError("Fit the model before requesting estimates")
        if self.transition_matrix_.shape[0] != source_categories:
            raise ValueError("Source categories differ from fitted data")
