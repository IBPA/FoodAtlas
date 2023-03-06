#!/bin/bash
set -e

root_dir=`pwd`

echo 'Downloading CTD.zip...'
curl -L https://ucdavis.box.com/shared/static/gqjox4fstsj7xruygv8q60446r2f06br --output CTD.zip

echo 'Downloading FDC.zip...'
curl -L https://ucdavis.box.com/shared/static/09zlzi1bgi5r37usojmxmif2rre7675t --output FDC.zip

echo 'Downloading FoodAtlas.zip...'
curl -L https://ucdavis.box.com/shared/static/y55utmxlehf6vcrhwomdab0xqn3qazzb --output FoodAtlas.zip

echo 'Downloading FooDB.zip...'
curl -L https://ucdavis.box.com/shared/static/1d1iznn5ow6eujd9i7tty8o6lstvrgbu --output FooDB.zip

echo 'Downloading Frida.zip...'
curl -L https://ucdavis.box.com/shared/static/79bbvtc7pxpyiehx7xao6d2qi05v3bn9 --output Frida.zip

echo 'Downloading Frida.zip...'
curl -L https://ucdavis.box.com/shared/static/79bbvtc7pxpyiehx7xao6d2qi05v3bn9 --output Frida.zip

echo 'Downloading MESH.zip...'
curl -L https://ucdavis.box.com/shared/static/vue7y9rvtdvuujo9y0x8oqw5dn81mh5b --output MESH.zip

echo 'Downloading NCBI_Taxonomy.zip...'
curl -L https://ucdavis.box.com/shared/static/zmplcwk6mtln838ktbf71c2t3nt2xiw2 --output NCBI_Taxonomy.zip

echo 'Downloading Phenol-Explorer.zip...'
curl -L https://ucdavis.box.com/shared/static/llcxnyuvh08yy7msd7jhtv5p2zbymquk --output Phenol-Explorer.zip

echo 'Downloading PubChem.zip...'
curl -L https://ucdavis.box.com/shared/static/qoku65gfiqsnkwwo9bej20i4qvyovam2 --output PubChem.zip

echo 'Unzipping all files...'
unzip "*.zip"

echo 'Deleting zipped files...'
rm *.zip
