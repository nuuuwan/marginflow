from dataclasses import dataclass


@dataclass(frozen=True)
class Party:
    """Describe an election category and its display color."""

    name: str
    color: str

    NAME_TO_COLOR = {
        "SLPP": "#8A1538",
        "NDF": "#00843D",
        "UPFA": "#0057A8",
        "NPP": "#7A263A",
        "SJB": "#F4C430",
        "Others": "#707070",
        "Rejected": "#929292",
        "Not Polled": "#B5B5B5",
        "New": "#D8D8D8",
    }

    @classmethod
    def from_name(cls, name):
        """Create a party using its official or category color.

        Args:
            name: Party or special category name.

        Returns:
            Party with a stable display color.
        """
        return cls(name, cls.NAME_TO_COLOR.get(name, "#707070"))

    def rgba(self, alpha):
        """Return the party color with the requested transparency.

        Args:
            alpha: Opacity between zero and one.

        Returns:
            Color formatted for Plotly links.
        """
        red, green, blue = bytes.fromhex(self.color.removeprefix("#"))
        return f"rgba({red}, {green}, {blue}, {alpha})"
