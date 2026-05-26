from extractor import CardDictExtractor, TextExtractor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.dummy import DummyClassifier

from scryfall.mtg_card_data import MTGCard


def train_dummy_pipeline(cards: list[MTGCard], colors: list[str]) -> Pipeline:
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "tfidf",
                            Pipeline(
                                [
                                    ("extract", TextExtractor()),
                                    ("tfidf", TfidfVectorizer()),
                                ]
                            ),
                        ),
                        (
                            "structured",
                            Pipeline(
                                [
                                    ("extract", CardDictExtractor()),
                                    ("vec", DictVectorizer(sparse=True)),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    )

    pipeline.fit(cards, colors)
    return pipeline
