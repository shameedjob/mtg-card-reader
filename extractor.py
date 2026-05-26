from sklearn.base import BaseEstimator, TransformerMixin
from scryfall.mtg_card_data import MTGCard


class TextExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: list[MTGCard]):
        return [
            " ".join([card.oracle_text, card.type_line, card.name])
            for card in X
        ]


class CardDictExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: list[MTGCard]):
        return [
            {
                "n_words": len(card.oracle_text.split()),
                "is_creature": "Creature" in card.type_line,
                "is_legendary": "Legendary" in card.type_line,
                "is_instant": "Instant" in card.type_line,
                "is_sorcery": "Sorcery" in card.type_line,
                "is_artifact": "Artifact" in card.type_line,
                "is_battle": "Battle" in card.type_line,
                "is_planeswalker": "Planeswalker" in card.type_line,
                "is_land": "Land" in card.type_line,
                "cmc": float(card.cmc)
            }
            for card in X
        ]
