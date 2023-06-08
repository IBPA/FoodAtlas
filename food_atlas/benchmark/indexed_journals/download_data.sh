#!/bin/bash
set -e

root_dir=`pwd`

echo 'Downloading umls.zip...'
curl -L https://ucdavis.box.com/shared/static/spesdtb0dik7y4ccava8lr0zfwssz0lr --output indexed_journals.zip

echo 'Unzipping all files...'
unzip "*.zip"

echo 'Deleting zipped files...'
rm *.zip
