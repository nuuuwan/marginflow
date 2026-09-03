from gig import Ent, EntType
from Party import Party
from SankeyDiagram import SankeyDiagram

from marginflow import MarginFlowModel


class ElectionTransitionMixin:
    """Fit, display, and plot transitions between two elections."""

    @staticmethod
    def add_new_columns(x1, x2):
        assert len(x1) == len(x2), "Both lists must have the same length."
        for row1, row2 in zip(x1, x2):
            total1, total2 = sum(row1), sum(row2)
            row1.append(max(0, total2 - total1))
            row2.append(max(0, total1 - total2))
        return x1, x2

    @classmethod
    def build_sankey(
        cls,
        model,
        source_counts,
        source_parties,
        target_parties,
        year1,
        year2,
    ):
        """Build the fitted transition Sankey and save it as a PNG."""
        return SankeyDiagram.build_sankey(
            model,
            source_counts,
            source_parties,
            target_parties,
            year1,
            year2,
            f"examples/lk-elections/transition-sankey-{year1}-to-{year2}.png",
            cls.P_LIMIT,
        )

    @classmethod
    def simulate_transition(cls, year1, year2):
        election1 = cls("presidential", year1)
        election2 = cls("presidential", year2)
        x1, x2 = cls.add_new_columns(election1.X, election2.X)
        source_parties = cls._parties(election1)
        target_parties = cls._parties(election2)
        cls._print_inputs(x1, x2, source_parties, target_parties)
        model = MarginFlowModel().fit(x1, x2)
        output_path = cls.build_sankey(
            model, x1, source_parties, target_parties, year1, year2
        )
        print(f"Wrote {output_path}")
        cls._print_transitions(model, source_parties, target_parties)

    @staticmethod
    def _parties(election):
        names = election._valid_parties + [
            "Others",
            "Rejected",
            "Not Polled",
            "New",
        ]
        return [Party.from_name(name) for name in names]

    @staticmethod
    def _print_inputs(x1, x2, source_parties, target_parties):
        first_pd = Ent.list_from_type(EntType.PD)[0]
        print(first_pd.id, first_pd.name)
        print([party.name for party in source_parties])
        print(x1[0])
        print([party.name for party in target_parties])
        print(x2[0])

    @classmethod
    def _print_transitions(cls, model, source_parties, target_parties):
        for source_index, source_party in enumerate(source_parties):
            for target_index, target_party in enumerate(target_parties):
                probability = model.transition_matrix_[
                    source_index, target_index
                ]
                if probability >= cls.P_LIMIT:
                    print(
                        f"{source_party.name} -> {target_party.name}: "
                        f"{probability:.1%}"
                    )
