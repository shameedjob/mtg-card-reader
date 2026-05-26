# randomly split data
import csv
from scryfall.mtg_card_data import MTGCard
from sklearn import model_selection
import random
random.seed = 32

def get_cards(file_path: str) -> list[MTGCard]:
    with open(file_path, "r") as file:
        lines = csv.DictReader(file, delimiter="\t")
        total_cards = [MTGCard.from_dict(l) for l in lines]
        return total_cards


# how do we split colors to be representative?
def split_data(mtg_cards: list[MTGCard]):
    colors = [
        random.choice(card.colors) for card in mtg_cards
    ]  # random color assignment

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        mtg_cards, colors, test_size=0.1, random_state=32
    )

    X_train, X_dev, y_train, y_dev = model_selection.train_test_split(
        X_train, y_train, test_size=1/9, random_state=32
    )
    return [X_train, y_train], [X_test, y_test], [X_dev, y_dev]


def get_training_data():
    cards = get_cards("data/processed/card_data.tsv")
    return split_data(cards)
