import os
import string

import pandas as pd
from Bio import Entrez
from tqdm import tqdm

Entrez.email = "fzli@ucdavis.edu"

STOPWORDS = [
    'a', 'about', 'again', 'all', 'almost', 'also', 'although', 'always',
    'among', 'an', 'and', 'another', 'any', 'are', 'as', 'at', 'be', 'because',
    'been', 'before', 'being', 'between', 'both', 'but', 'by', 'can', 'could',
    'did', 'do', 'does', 'done', 'due', 'during', 'each', 'either', 'enough',
    'especially', 'etc', 'for', 'found', 'from', 'further', 'had', 'has',
    'have', 'having', 'here', 'how', 'however', 'i', 'if', 'in', 'into', 'is',
    'it', 'its', 'itself', 'just', 'kg', 'km', 'made', 'mainly', 'make', 'may',
    'mg', 'might', 'ml', 'mm', 'most', 'mostly', 'must', 'nearly', 'neither',
    'no', 'nor', 'obtained', 'of', 'often', 'on', 'our', 'overall', 'perhaps',
    'pmid', 'quite', 'rather', 'really', 'regarding', 'seem', 'seen',
    'several', 'should', 'show', 'showed', 'shown', 'shows', 'significantly',
    'since', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs',
    'them', 'then', 'there', 'therefore', 'these', 'they', 'this', 'those',
    'through', 'thus', 'to', 'upon', 'various', 'very', 'was', 'we', 'were',
    'what', 'when', 'which', 'while', 'with', 'within', 'without', 'would',
]


def format_query(title):
    query = ' AND '.join(
        [f"{x}[Title]" for x in title.split(' ') if x not in STOPWORDS]
    )

    return query


def query_pmid(query):
    with Entrez.esearch(db='pubmed', term=query) as handle:
        record = Entrez.read(handle)

    return record['IdList']


def query_titles(pmids):
    entries = []
    with Entrez.esummary(db='pubmed', id=pmids) as handle:
        records = Entrez.parse(handle)
        for record in records:
            entries += [{
                'pmid': record['Id'],
                'title': record['Title'],
            }]

    return pd.DataFrame(entries)


def get_pmids(titles):
    if not os.path.exists("data/_CACHED_QUERIES/_title_to_pmid.csv"):
        hash_table_title2pmid = pd.DataFrame()
    else:
        hash_table_title2pmid = pd.read_csv(
            "data/_CACHED_QUERIES/_title_to_pmid.csv",
            index_col='query',
        )

    if not os.path.exists("data/_CACHED_QUERIES/_pmid_to_title.csv"):
        hash_table_pmid2title = pd.DataFrame()
    else:
        hash_table_pmid2title = pd.read_csv(
            "data/_CACHED_QUERIES/_pmid_to_title.csv",
            index_col='pmid',
        )

    # Hashing all the title-pmid pairs.
    hash_entries_new = []
    failed = False
    try:
        for title in tqdm(titles):
            title_query = format_query(title.lower())

            # If the food name is already in the hash table, retrieve it.
            try:
                hash_table_title2pmid.loc[title_query]['pmid']
                continue
            except KeyError:
                pass

            pmid = ';'.join(query_pmid(title_query))
            hash_entries_new += [{
                'query': title_query,
                'pmid': pmid
            }]
    except Exception as e:
        failed = True
        print("The following error occupied. Saving the hash table.")
        print("==============================================")
        print(type(e).__name__)
        print(e)
        print("==============================================")

    if hash_entries_new:
        hash_table_title2pmid = pd.concat(
            [
                hash_table_title2pmid,
                pd.DataFrame(hash_entries_new).set_index('query'),
            ],
        ).sort_index()
        hash_table_title2pmid = hash_table_title2pmid.reset_index()\
            .drop_duplicates('query', keep='first').set_index('query')
        hash_table_title2pmid.to_csv(
            "data/_CACHED_QUERIES/_title_to_pmid.csv"
        )

    if failed:
        print("Exiting the program.")
        exit()

    # Verify that the pmids are correct.
    # First, get all the titles with PMIDs.
    pmids_new = []
    for title in titles:
        title_query = format_query(title.lower())
        if pd.notna(hash_table_title2pmid.loc[title_query]['pmid']):
            for pmid in hash_table_title2pmid.loc[title_query]['pmid']\
                    .split(';'):
                try:
                    hash_table_pmid2title.loc[int(pmid)]
                    continue
                except KeyError:
                    pmids_new += [pmid]
    if pmids_new:
        hash_table_pmid2title_new = query_titles(pmids_new).astype(
            {'pmid': 'Int64'})
        hash_table_pmid2title = pd.concat(
            [
                hash_table_pmid2title,
                hash_table_pmid2title_new.set_index('pmid'),
            ],
        ).sort_index()
        hash_table_pmid2title = hash_table_pmid2title.reset_index()\
            .drop_duplicates('pmid', keep='first').set_index('pmid')
        hash_table_pmid2title.to_csv(
            "data/_CACHED_QUERIES/_pmid_to_title.csv"
        )

    # Start doing title matching.
    def clean_title(title):
        """Lowercase and remove certain punctuations."""
        title = title.lower()

        chars_to_remove = list(string.punctuation) \
            + [' ', u'\u2013', u'\u2014']
        for char in chars_to_remove:
            title = title.replace(char, '')

        return title

    pmids_verified = []
    for i, title in enumerate(titles):
        pmids = hash_table_title2pmid.loc[format_query(title.lower())]['pmid']

        if pd.isna(pmids):
            pmids_verified += ['']
            continue

        matched = False
        for pmid in pmids.split(';'):
            title_pubmed = hash_table_pmid2title.loc[int(pmid)]['title']
            if clean_title(title) == clean_title(title_pubmed):
                pmids_verified += [pmid]
                matched = True
                break

        if not matched:
            pmids_verified += ['']

            # Print out the mismatched titles for manual inspection.
            for pmid in pmids.split(';'):
                title_pubmed = hash_table_pmid2title.loc[int(pmid)]['title']
                print(f"DB    : {title}")
                print(f"PubMed: {title_pubmed}")
                print(f"PMID  : {pmid}")
                print()
            print()

    return pmids_verified
