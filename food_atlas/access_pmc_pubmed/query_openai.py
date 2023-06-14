import argparse
import os
from pathlib import Path
import pickle
import sys

import openai
from tqdm import tqdm
import pandas as pd

from secret import OPENAI_API_KEY, ORGANIZATION_ID

CHAT_PRICING = 0.002  # $0.002 / 1k tokens
COMPLETION_PRICING = 0.02  # $0.02 / 1k tokens

openai.api_key = OPENAI_API_KEY
openai.organization = ORGANIZATION_ID

PRE_ANNOTATE_INSTRUCTION = \
    'Return True if the following sentence contains food-chemical relationships, ' \
    'and False otherwise.'

TUPLE_EXTRACT_INSTRUCTION = \
    'Extract tuples of format (food, food part, chemical, chemical concentration). ' \
    'Leave food part and/or chemical concentration blank if nothing found. ' \
    'Return one tuple per line. Return only "None" if no tuples are found.'

MODE_INSTRUCTION_DICT = {
    'pre-annotate': PRE_ANNOTATE_INSTRUCTION,
    'tuple-extract': TUPLE_EXTRACT_INSTRUCTION,
}


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        help='pre-annotate|tuple-extract',
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='.tsv file containing one premise for each line.',
    )

    parser.add_argument(
        '--endpoint',
        type=str,
        required=True,
        help='OpenAI endpoint to use (chat|completion)',
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file.',
    )

    args = parser.parse_args()

    # check arguments
    assert args.mode in ['pre-annotate', 'tuple-extract']
    assert args.endpoint in ['chat', 'completion']

    if args.mode == 'pre-annotate':
        assert args.endpoint == 'completion'

    return args


def parse_input(filepath):
    with open(filepath, 'rb') as _f:
        df = pickle.load(_f)

    df.reset_index(inplace=True, drop=True)

    return df


def main():
    args = parse_argument()

    df = parse_input(args.input)

    if args.endpoint == 'chat':
        price = CHAT_PRICING
    elif args.endpoint == 'completion':
        price = COMPLETION_PRICING

    if args.mode == 'pre-annotate':
        num_positives = 0

    rows = []
    num_tokens = []
    pbar = tqdm(df.iterrows(), total=df.shape[0])

    for idx, row in pbar:
        # calculate cost
        total_tokens = sum(num_tokens)
        cost = (total_tokens / 1000) * price
        if idx == 0:
            approx_total_cost = 0
        else:
            approx_total_cost = (df.shape[0] / idx) * cost

        pbar.set_description(
            f'Total {total_tokens} tokens (${cost:.2f}) used. '
            f'Approximated total cost is ${approx_total_cost:.2f}.'
        )

        if args.endpoint == 'chat':
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                temperature=0,
                messages=[
                    {'role': 'system', 'content': MODE_INSTRUCTION_DICT[args.mode]},
                    {'role': 'user', 'content': row['text']}
                ],
            )
        elif args.endpoint == 'completion':
            if args.mode == 'tuple-extract':
                max_tokens = 4096
            elif args.mode == 'pre-annotate':
                max_tokens = 3

            response = openai.Completion.create(
                model='text-davinci-003',
                temperature=0,
                max_tokens=max_tokens,
                prompt=MODE_INSTRUCTION_DICT[args.mode] + '\n' + row['text'],
            )

        row['response'] = response
        rows.append(row)

        # debug
        if True:
            print(row['text'])
            if args.endpoint == 'chat':
                print(response['choices'][0]['message']['content'].strip())
            elif args.endpoint == 'completion':
                print(response['choices'][0]['text'].strip())

        num_tokens.append(response['usage']['total_tokens'])

        if args.mode == 'pre-annotate':
            answer = response['choices'][0]['text'].strip().lower()
            if answer == 'true':
                num_positives += 1
                if num_positives == 5000:
                    break

                print(f'Number of positives so far: {num_positives}')

    df = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, 'wb') as _f:
        pickle.dump(df, _f)


if __name__ == '__main__':
    main()
