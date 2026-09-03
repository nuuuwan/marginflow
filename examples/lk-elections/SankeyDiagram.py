from pathlib import Path

import numpy as np
import plotly.graph_objects as go


class SankeyDiagram:
    """Render estimated category movements as a Sankey diagram."""

    @classmethod
    def build_sankey(
        cls,
        model,
        source_counts,
        source_parties,
        target_parties,
        source_year,
        target_year,
        output_path,
        probability_limit=0.05,
    ):
        """Build a Sankey diagram and save it as a PNG image.

        Args:
            model: Fitted MarginFlow model.
            source_counts: Starting counts arranged by group and category.
            source_parties: Starting parties and their colors.
            target_parties: Final parties and their colors.
            source_year: Year shown beside source labels.
            target_year: Year shown beside target labels.
            output_path: Location of the PNG file to create.
            probability_limit: Smallest transition probability to draw.

        Returns:
            Path to the generated PNG file.
        """
        source, target, value, color = cls._links(
            model, source_counts, source_parties, probability_limit
        )
        node = cls._nodes(
            source_parties, target_parties, source_year, target_year
        )
        figure = go.Figure(
            go.Sankey(
                arrangement="snap",
                domain={"y": [0.0, 0.9]},
                node=node,
                link={
                    "source": source,
                    "target": target,
                    "value": value,
                    "color": color,
                },
            )
        )
        figure.update_layout(
            title_text=f"Estimated transitions: {source_year} to {target_year}",
            width=1600,
            height=max(
                900, 70 * max(len(source_parties), len(target_parties))
            ),
            font_size=13,
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_image(path, format="png", scale=2)
        return path

    @staticmethod
    def _nodes(source_parties, target_parties, source_year, target_year):
        parties = source_parties + target_parties
        labels = [f"{party.name} ({source_year})" for party in source_parties]
        labels += [f"{party.name} ({target_year})" for party in target_parties]
        return {
            "label": labels,
            "color": [party.color for party in parties],
            "x": [0.01] * len(source_parties) + [0.99] * len(target_parties),
            "pad": 30,
            "thickness": 22,
        }

    @staticmethod
    def _links(model, source_counts, source_parties, probability_limit):
        probabilities = model.transition_matrix_
        flows = model.aggregate_flows(source_counts)
        target_offset = probabilities.shape[0]
        sources, targets, values, colors = [], [], [], []
        for source_index, target_index in np.ndindex(probabilities.shape):
            if probabilities[source_index, target_index] < probability_limit:
                continue
            sources.append(source_index)
            targets.append(target_offset + target_index)
            values.append(flows[source_index, target_index])
            colors.append(source_parties[source_index].rgba(0.45))
        return sources, targets, values, colors
