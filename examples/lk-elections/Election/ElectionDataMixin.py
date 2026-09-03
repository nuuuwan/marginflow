from functools import cached_property

from gig import Ent, EntType, GIGTable


class ElectionDataMixin:
    """Load election results and arrange them as a count matrix."""

    @cached_property
    def gig_table(self) -> GIGTable:
        return GIGTable(
            f"government-elections-{self.election_type}",
            "regions-ec",
            str(self.election_year),
        )

    @cached_property
    def _valid_parties(self) -> list[str]:
        national_result = Ent.from_id("LK").gig(self.gig_table)
        party_to_votes = self._party_votes(national_result.dict)
        limit = sum(party_to_votes.values()) * self.P_LIMIT
        return [
            party for party, votes in party_to_votes.items() if votes >= limit
        ]

    @cached_property
    def X(self) -> list[list[int]]:
        return [self._matrix_row(pd) for pd in Ent.list_from_type(EntType.PD)]

    def _matrix_row(self, pd):
        stats = pd.gig(self.gig_table).dict
        party_to_votes = self._party_votes(stats)
        valid_votes = [
            party_to_votes.get(party, 0) for party in self._valid_parties
        ]
        others = stats["valid"] - sum(valid_votes)
        not_polled = stats["electors"] - stats["polled"]
        return valid_votes + [others, stats["rejected"], not_polled]

    @staticmethod
    def _party_votes(stats):
        totals = {"electors", "polled", "valid", "rejected"}
        return {
            key: value for key, value in stats.items() if key not in totals
        }
