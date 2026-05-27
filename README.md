# MTG Card Color Classifier

A logistic regression classifier that predicts the color identity of Magic: The Gathering cards from card text and structured features.

## Overview

Cards are represented using a combination of TF-IDF features (oracle text, type line, name) and structured card attributes. A multi-label logistic regression pipeline predicts color identity across the six color classes: W, U, B, R, G, and N (colorless).

## Project Structure

```
mtg-card-reader/
├── data/               # Card data (TSV)
├── scryfall/           # Scryfall data loading and card schema
├── extractor.py        # sklearn transformers (TextExtractor, CardDictExtractor)
├── split.py            # Train/test split utilities
├── train.py            # Trains the logistic regression pipeline
├── baseline.py         # Majority-class dummy baseline
├── evaluate.py         # Evaluation on held-out data
├── test.py             # Predict a card's color identity from command-line parameters
└── requirements.txt
```

## Setup

```bash
conda env create -f environment.yml
```

## Usage

**Evaluate model performance**
```bash
python evaluate.py [options]
```

Trains the pipeline on the training split and reports Micro, Macro, and Sample Average accuracy for the logistic regression model, a thresholded variant, and the dummy baseline.

| Flag | Long form | Description |
|------|-----------|-------------|
| `-d` | `--dev` | Evaluate on the dev set instead of test |
| `-t` | `--training_size` | Limit training data to N examples |
| `-ratio` | `--ratio` | Probability threshold for color assignment (default: `0.1667`) |
| `-r` | `--result` | Write per-card predictions to a TSV file |
| `-f` | `--features` | Write top model features to a file |

Example:
```bash
python evaluate.py --dev --ratio 0.2 --result predictions.tsv
```

---

**Predict a card's color identity**
```bash
python test.py -n "Card Name" -c <cmc> -d "Oracle text" -t "Type line"
```

| Flag | Long form | Description |
|------|-----------|-------------|
| `-n` | `--name` | Card name |
| `-c` | `--cmc` | Converted mana cost (numeric) |
| `-d` | `--oracle_text` | Oracle (rules) text |
| `-t` | `--type_line` | Type line (e.g. `Instant`, `Creature — Elf`) |

Examples:
```bash
python test.py -n "Lightning Bolt" -c 1 -d "Lightning Bolt deals 3 damage to any target." -t "Instant"
```

```bash
python test.py -n 'Bella the Bold' -c 4.0 -d 'Deal damage to chosen creature with flying.' -t 'Creature'
# ['G', 'R', 'W']
```

Output is the predicted color identity labels: `W`, `U`, `B`, `R`, `G`, or `N` (colorless). Multi-color cards return a list.

## Model

The pipeline uses a `FeatureUnion` of two sub-pipelines:

- **TF-IDF** — unigram/bigram TF-IDF over concatenated oracle text, type line, and card name (up to 10k features)
- **Structured** — dict-vectorized card attributes, standard-scaled

Both are fed into a `LogisticRegression` classifier. Multi-color cards are handled by thresholding per-class probabilities against a uniform prior (1/6).

## Data

Card data sourced from [Scryfall](https://scryfall.com/docs/api) via the bulk-data endpoint (`[https://api.scryfall.com/bulk-data/](https://api.scryfall.com/bulk-data/e2ef41e3-5778-4bc2-af3f-78eca4dd9c23)`). Multi-color cards are assigned a single color for training via random sampling.

**Preparing data from Scryfall**

After downloading the bulk JSON from the Scryfall bulk-data endpoint, save it to `data/raw/card-data.json`. Then convert it to the TSV format used by the classifier:

```bash
python scryfall/scryfall-card-reader.py -f <field1> <field2> ... -o <output.tsv>
```

| Flag | Long form | Description |
|------|-----------|-------------|
| `-f` | `--fields` | Scryfall JSON fields to extract (e.g. `name oracle_text type_line cmc color_identity`) |
| `-c` | `--count` | Max number of cards to export (default: all) |
| `-o` | `--output` | Output TSV file path (prints to stdout if omitted) |

The script deduplicates cards by `oracle_id`, extracts the requested fields, and writes a tab-separated file with a header row. Empty field values are replaced with `[NONE]`.

Example (extracting the fields needed by the classifier):
```bash
python scryfall/scryfall-card-reader.py -f name oracle_text type_line cmc color_identity -o data/cards.tsv
```

## Results

| Model    | Metric | Score |
|----------|--------|-------|
| Baseline | ...    | ...   |
| LogReg   | ...    | ...   |
