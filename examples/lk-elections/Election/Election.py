from dataclasses import dataclass

from Election.ElectionDataMixin import ElectionDataMixin
from Election.ElectionTransitionMixin import ElectionTransitionMixin


@dataclass
class Election(ElectionDataMixin, ElectionTransitionMixin):
    """Provide grouped results and transitions for one election."""

    election_type: str
    election_year: int

    P_LIMIT = 0.05
