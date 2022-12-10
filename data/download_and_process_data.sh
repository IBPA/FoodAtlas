#!/bin/bash

# exit immediately on error
set -e

root_dir=`pwd`

# Download and prepare FooDB
mkdir -p FooDB
cd FooDB
wget https://foodb.ca/public/system/downloads/foodb_2020_4_7_csv.tar.gz
tar -xf foodb_2020_4_7_csv.tar.gz
rm foodb_2020_4_7_csv.tar.gz

python process_foodb.py \
    --food_filepath=./foodb_2020_04_07_csv/Food.csv \
    --output_filepath=./foodb_foods.txt

# NCBI Taxonomy
# Downloaded new_taxdump.zip from https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/ on 11/30/2022 at 9:53 AM.
