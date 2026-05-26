import argparse
import joblib
import evaluate

from sklearn.pipeline import Pipeline
from scryfall.mtg_card_data import MTGCard
def test_card(pipeline:Pipeline, card:MTGCard):
    prediction = pipeline.predict_proba([card])
    # print(prediction)
    return evaluate.get_class_from_prediction(prediction[0])

def main(args):
    pipeline:Pipeline = joblib.load('data/pipeline/pipeline.joblib')
    card_dict = {
        'color_identity': '[NONE]',
        'name': args.name,
        'oracle_text': args.oracle_text,
        'cmc': args.cmc,
        'type_line': args.type_line
    }
    print(test_card(pipeline, MTGCard.from_dict(card_dict)))
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name', type=str)
    parser.add_argument('-c','--cmc', type=float)
    parser.add_argument('-d','--oracle_text', type=str)
    parser.add_argument('-t','--type_line', type=str)
    # parser.add_argument('-','--name')
    args = parser.parse_args()
    main(args)
    