import argparse

import pandas as pd
from pandarallel import pandarallel

from common_utils.knowledge_graph import KnowledgeGraph


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add link prediction results.")

    parser.add_argument(
        "--input_filepath",
        type=str,
        required=True,
        help="Input filepath."
    )

    parser.add_argument(
        "--input_kg_dir",
        type=str,
        required=True,
        help="KG directory to merge the MESH to.",
    )

    parser.add_argument(
        "--output_kg_dir",
        type=str,
        required=True,
        help="KG directory to merge the MESH to.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Are you adding annotated positives of predictions (annotated|predicted).",
    )

    parser.add_argument(
        "--nb_workers",
        type=int,
        help="Number of workers for pandarallel.",
    )

    args = parser.parse_args()
    return args


def _is_true(row):
    result = True
    prob_names = [x for x in row.index.values.tolist() if x.startswith('pred_run_')]
    for x in prob_names:
        result &= row[x]
    return result


def main():
    args = parse_argument()

    if args.nb_workers is None:
        pandarallel.initialize(progress_bar=True)
    else:
        pandarallel.initialize(progress_bar=True, nb_workers=args.nb_workers)

    # read data
    if args.mode == "predictions":
        df = pd.read_csv(args.input_filepath, sep='\t')
        print(df)

        df['pred'] = df.parallel_apply(lambda row: _is_true(row), axis=1)
        df_pos = df[df['pred']]
        df_pos = df_pos[['head', 'relation', 'tail', 'prob_mean', 'prob_std']]
        print(df_pos)
    elif args.mode == "validations":
        raise NotImplementedError()
    else:
        raise ValueError()

    df_pos['source'] = 'FoodAtlas:prediction:lp'
    df_pos['quality'] = 'low'
    print(f"Original positive triples shape: {df_pos.shape[0]}")

    # add predictions
    fa_kg = KnowledgeGraph(kg_dir=args.input_kg_dir, nb_workers=args.nb_workers)
    fa_kg.add_triples(df_pos, source='lp')
    fa_kg.save(kg_dir=args.output_kg_dir)


if __name__ == '__main__':
    main()
