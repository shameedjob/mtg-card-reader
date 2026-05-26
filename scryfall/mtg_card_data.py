from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MTGCard:
    name: str
    oracle_text: str
    type_line: str
    colors: list[str]
    cmc: float
    @classmethod
    def from_dict(cls, data: dict) -> "MTGCard":
        return cls(
            name=data["name"],
            oracle_text=data["oracle_text"],
            type_line=data["type_line"],
            cmc = data['cmc'],
            colors=(
                list(data["color_identity"])
                if data["color_identity"] != "[NONE]"
                else ["N"]
            ),
        )
