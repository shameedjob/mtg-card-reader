# compute performance on held-out data
import split as split
import train as train
import baseline as baseline
import numpy
import argparse
from sklearn.pipeline import Pipeline


def convert_predictions(values: list[float]) -> dict:
    
    return {
        "B": values[0],
        "G": values[1],
        "N": values[2],
        "R": values[3],
        "U": values[4],
        "W": values[5],
    }


def get_class_from_prediction(values: list[float], ratio:float=1/6) -> list[str]:
    converted = convert_predictions(values)
    predicted_value = []
    
    for l in converted:
        if converted[l] > ratio:
            predicted_value.append(l)
    if 'N' in predicted_value:
        predicted_value = ['N']
    return predicted_value

def get_class_from_prediction_thresh(values: list[float], ratio:float=1/6) -> list[str]:
    converted = convert_predictions(values)
    predicted_value = []
    n_value = converted['N']
    for l in converted:
        if converted[l] > n_value:
            predicted_value.append(l)
    if len(predicted_value) < 1:
        predicted_value = ['N']
    return predicted_value


def top_features(pipeline: Pipeline, n: int = 20, output:str|None = None):
    tfidf_names = (
        pipeline.named_steps["features"]
        .transformer_list[0][1]
        .named_steps["tfidf"]
        .get_feature_names_out()
    )
    dict_names = (
        pipeline.named_steps["features"]
        .transformer_list[1][1]
        .named_steps["vec"]
        .get_feature_names_out()
    )
    feature_names = numpy.concatenate([tfidf_names, dict_names])
    n_tfidf = len(tfidf_names)

    clf = pipeline.named_steps["clf"]
    classes = clf.classes_
    if output:
        output_file = open(output, 'w')
    
    for i, cls in enumerate(classes):
        coefs = clf.coef_[i]
        print(f"\n--- {cls} ---", file=output_file)

        # top n tfidf features by magnitude
        tfidf_coefs = coefs[:n_tfidf]
        top_tfidf_idx = numpy.argsort(numpy.abs(tfidf_coefs))[::-1][:n]
        print(f"  [tfidf top {n}]", file=output_file)
        for idx in top_tfidf_idx:
            print(f"    {feature_names[idx]:40s} {tfidf_coefs[idx]:+.4f}", file=output_file)

        # all structured features
        structured_coefs = coefs[n_tfidf:]
        print(f"  [structured all]", file=output_file)
        ranked_struct = numpy.argsort(numpy.abs(structured_coefs))[::-1]
        for idx in ranked_struct:
            print(f"    {dict_names[idx]:40s} {structured_coefs[idx]:+.4f}", file=output_file)


def get_averages(
    predictions, true_colors, sampled_colors, method:callable, output: str | None = None
):

    totals: list[int] = []
    correct: list[int] = []
    sampled_success = 0
    if output:
        output_file = open(output, "w")
        print(
            "predicted",
            "gold",
            sep='\t',
            file=output_file
        )

    for predicted, gold_colors, sampled_true in zip(
        predictions, true_colors, sampled_colors
    ):
        pred_colors = method(predicted, args.ratio)
        if sampled_true in pred_colors:
            sampled_success += 1

        correct_count = 0
        # print(pred_colors)
        for p in pred_colors:
            if p in gold_colors:
                correct_count += 1
        if output:
            print(''.join(pred_colors), ''.join(gold_colors), sep="\t", file=output_file)

        totals.append(max(len(gold_colors), len(pred_colors)))
        correct.append(correct_count)

    macro = sum(correct) / sum(totals)
    micro = sum([c / t for c, t in zip(correct, totals)]) / len(totals)
    sampled_avg = sampled_success / len(totals)
    return macro, micro, sampled_avg


def main(args):
    train_values, test_values, dev_values = split.get_training_data()

    values_to_check = dev_values if args.dev else test_values

    if args.training_size > -1:
        train_values = (
            train_values[0][: args.training_size],
            train_values[1][: args.training_size],
        )
    pipeline = train.train_pipeline(train_values[0], train_values[1])
    dummy_pipeline = baseline.train_dummy_pipeline(
        train_values[0], train_values[1]
    )

    true_colors = [c.colors for c in values_to_check[0]]
    pipe_predictions = pipeline.predict_proba(values_to_check[0])
    dummy_predictions = dummy_pipeline.predict_proba(values_to_check[0])

    micro, macro, sampled_avg = get_averages(
        pipe_predictions, true_colors, values_to_check[1], get_class_from_prediction,  output=args.result
    )
    print("Model")
    print(
        f"Micro: {micro:.4f}\nMacro: {macro:.4f}\nSample Avg: {sampled_avg:.4f}\n"
    )

    micro, macro, sampled_avg = get_averages(
        pipe_predictions, true_colors, values_to_check[1], get_class_from_prediction_thresh
    )
    print("Model B")
    print(
        f"Micro: {micro:.4f}\nMacro: {macro:.4f}\nSample Avg: {sampled_avg:.4f}\n"
    )

    micro, macro, sampled_avg = get_averages(
        dummy_predictions, true_colors, values_to_check[1], get_class_from_prediction
    )
    print("Dummy")
    print(
        f"Micro: {micro:.4f}\nMacro: {macro:.4f}\nSample Avg: {sampled_avg:.4f}"
    )

    top_features(pipeline, output=args.features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result", type=str, default=None)
    parser.add_argument("-f", "--features", type=str, default=None)
    parser.add_argument("-t", "--training_size", type=int, default=-1)
    parser.add_argument("-d", "--dev", action="store_true")
    parser.add_argument("-ratio", "--ratio", default=1/6, type=float)

    args = parser.parse_args()
    main(args)
