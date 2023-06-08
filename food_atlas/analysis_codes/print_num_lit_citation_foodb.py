import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv(
        "data/FooDB/foodb_2020_04_07_csv/Content.csv",
        low_memory=False,
    )

    # Count citations with given types.
    counts = data[
        ['citation', 'citation_type']
    ].value_counts().reset_index().rename(columns={0: 'count'})
    counts.to_csv("foodb_ref.csv")
    n_total = counts['count'].sum()
    n_excl = 0  # Number of citations to exclude due to are not scientific lit.
    n_excl += counts.query(
        "citation_type.isin(['PREDICTED', 'DATABASE'])")['count'].sum()
    n_excl += counts.query(
        "~(citation_type.isin(['PREDICTED', 'DATABASE'])) "
        "& (citation == 'MANUAL')"
    )['count'].sum()

    # Number of citations with scientific literature.
    n_lit = n_total - n_excl

    print(n_lit / n_total)  # 0.003992 = 0.4%
