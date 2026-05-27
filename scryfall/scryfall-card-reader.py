#!/usr/bin/env python3
import argparse

# from analysis import mtg_card_data
import json
import logging


def process(fields: list[str], count: int = -1, output: str | None = None):
    with open(args.input, "r") as file:
        file.readline()
        index = 0
        imported_cards = set()
        card_file = None if output == None else open(output, "w")
        print(*fields, sep="\t", file=card_file)
        total, exported = 0, 0
        while line := file.readline():
            if count == 0:
                break
            try:
                card_data = json.loads(line[:-2])
                if card_data["oracle_id"] in imported_cards: continue
                imported_cards.add(card_data["oracle_id"])
                count -= 1

                index += 1
                total += 1
                
                field_values = [
                    (
                        "".join(card_data[field])
                        .replace("\n", " ")
                        .replace("\t", " ")
                        if type(card_data[field]) == str or type(card_data[field]) == list
                    else card_data[field]) for field in fields]
                for i, data in enumerate(field_values):
                    if data == "":
                        field_values[i] = "[NONE]"

                print(*field_values, sep="\t", file=card_file)
                exported += 1
            except:
                pass
                # logging.exception(f"index: {index}")
        print(f"{exported} Cards Imported| {total-exported} failed to export")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Card selector")
    parser.add_argument("-f", "--fields", nargs="+", default=['name', 'oracle_text', 'color_identity', 'type_line', 'cmc'])
    parser.add_argument("-c", "--count", default=-1, type=int)
    parser.add_argument("-i", "--input", type=str, default='data/raw/card-data.json')
    parser.add_argument("-o", "--output", default=None, type=str)
    args = parser.parse_args()
    process(args.fields, args.count, args.output)
