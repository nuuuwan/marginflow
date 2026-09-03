from dataclasses import dataclass
from functools import cached_property

from gig import Ent, EntType, GIGTable

from marginflow import MarginFlowModel


@dataclass
class Election:
    election_type: str
    election_year: int

    P_LIMIT = 0.05

    @cached_property
    def gig_table(self) -> GIGTable:
        return GIGTable(
            f"government-elections-{self.election_type}",
            "regions-ec",
            str(self.election_year),
        )

    @cached_property
    def _valid_parties(self) -> list[str]:
        ent_lk = Ent.from_id("LK")
        national_result = ent_lk.gig(self.gig_table)
        party_to_stats = national_result.dict
        party_to_votes = {
            k: v
            for k, v in party_to_stats.items()
            if k not in {"electors", "polled", "valid", "rejected"}
        }
        total = sum(party_to_votes.values())
        limit = total * self.P_LIMIT
        party_to_votes = {
            k: v for k, v in party_to_votes.items() if v >= limit
        }
        return list(party_to_votes.keys())

    @cached_property
    def X(self) -> list[list[int]]:

        ent_pds = Ent.list_from_type(EntType.PD)
        valid_parties = self._valid_parties

        def get_matrix_row(pd):
            pd_result = pd.gig(self.gig_table)
            party_to_stats = pd_result.dict
            electors = party_to_stats["electors"]
            valid = party_to_stats["valid"]
            rejected = party_to_stats["rejected"]
            polled = party_to_stats["polled"]
            party_to_votes = {
                k: v
                for k, v in party_to_stats.items()
                if k not in {"electors", "polled", "valid", "rejected"}
            }
            valid_party_to_votes = {
                k: v for k, v in party_to_votes.items() if k in valid_parties
            }
            total_valid_votes = sum(valid_party_to_votes.values())
            others_votes = valid - total_valid_votes
            not_polled = electors - polled
            return list(valid_party_to_votes.values()) + [
                others_votes,
                rejected,
                not_polled,
            ]

        return [get_matrix_row(pd) for pd in ent_pds]

    @staticmethod
    def add_new_columns(x1, x2):
        assert len(x1) == len(x2), "Both lists must have the same length."
        new_x1 = []
        new_x2 = []
        for x1i, x2i in zip(x1, x2):
            sum_x1 = sum(x1i)
            sum_x2 = sum(x2i)
            new_value1 = max(0, sum_x2 - sum_x1)
            new_value2 = max(0, sum_x1 - sum_x2)
            x1i.append(new_value1)
            x2i.append(new_value2)

            new_x1.append(x1i)
            new_x2.append(x2i)

        return new_x1, new_x2


if __name__ == "__main__":
    election1 = Election("presidential", 2015)
    election2 = Election("presidential", 2019)

    X1 = election1.X
    X2 = election2.X

    X1, X2 = Election.add_new_columns(X1, X2)

    model = MarginFlowModel().fit(X1, X2)

    for i1, party1 in enumerate(
        election1._valid_parties + ["Others", "Rejected", "Not Polled", "New"]
    ):
        for i2, party2 in enumerate(
            election2._valid_parties
            + ["Others", "Rejected", "Not Polled", "New"]
        ):
            p = model.transition_matrix_[i1, i2]
            if p < Election.P_LIMIT:
                continue
            print(f"{party1} -> {party2}: {p:.1%}")
