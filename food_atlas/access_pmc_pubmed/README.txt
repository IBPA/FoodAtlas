###########
# ROUTE 1 #
###########

* Search PubMed & PMC using the query keywords.

This searches the queries in {query} file on the PubMed and/or PMC,
saves the results at {query_uid_results_filepath}.

python search_pubmed_pmc.py \
    --query=../../dataFoodAtlas/queries.txt \
    --db=both \
    --query_uid_results_filepath=../../outputs/pmc_pubmed/query_uid_results.tsv


###########
# ROUTE 2 #
###########

* Reformat the LitSense query results (from the previous pipeline).

Previous query results were in JSON files for each query keyword.
We reformat it and save it as a single pickle file of DataFrame structure.

python parse_litsense_to_chatgpt_input.py

Intpu: ../../data/FoodAtlas/litsense_query/queries_output/*
Output: ../../data/FoodAtlas/premises_from_litsense.pkl


* (Optional) Filter out the data from the above that will be used for premise classifier.

The above data contains lots of premises that do not contain any
food-chemical relationships. Instead of feeding them all through OpenAI GPT,
which would cost a lot of money, we filter out any premises that potentially
do not contain any food-chemical relationships by keeping only the premises
that contain both 'species' and 'chemical' entities.

python generate_data_for_premise_classifier.py \
    --input=../../data/FoodAtlas/premises_from_litsense.pkl \
    --output=../../data/FoodAtlas/premises_from_litsense_filtered.pkl

* (Optional) Further filter out the above data using ChatGPT to generate 'pre-annotate' data.

The above filtered data contains premises that contain both 'species' and 'chemical' entities.
But this does not necessarily mean they contain food-chemical relationships. Instead of
annotating everything ourself, we use OpenAI GPT to pre-annotate the premises.

python query_openai.py \
    --mode=pre-annotate \
    --input=../../data/FoodAtlas/premises_from_litsense_small_filtered.pkl \
    --endpoint=completion \
    --output=../../data/FoodAtlas/premises_from_litsense_small_filtered_pre_annotated.pkl
