import os

from extractor import TextExtractor, CardDictExtractor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scryfall.mtg_card_data import MTGCard

import joblib



def train_pipeline(cards: list[MTGCard], colors: list[str]) -> Pipeline:
    tfidf_pipeline = Pipeline(
        [
            ("extract", TextExtractor()),
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=10_000,
                ),
            ),
        ]
    )

    structured_pipeline = Pipeline(
        [
            ("extract", CardDictExtractor()),
            ("vec", DictVectorizer(sparse=True)),
            ("scale", StandardScaler(with_mean=False))
        ]
    )

    features = FeatureUnion(
        [
            ("tfidf", tfidf_pipeline),
            ("structured", structured_pipeline),
        ]
    )

    clf = LogisticRegression(max_iter=10_000, class_weight="balanced")

    pipeline = Pipeline(
        [
            ("features", features),
            ("clf", clf),
        ]
    )

    pipeline.fit(cards, colors)
    os.makedirs('data/pipeline/', exist_ok=True)
    joblib.dump(pipeline, 'data/pipeline/pipeline.joblib')
    return pipeline
    # pipeline.transform
