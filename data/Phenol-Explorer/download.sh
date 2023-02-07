#!/bin/bash

wget http://phenol-explorer.eu/system/downloads/current/composition-data.xlsx.zip
wget http://phenol-explorer.eu/system/downloads/current/compounds.csv.zip
wget http://phenol-explorer.eu/system/downloads/current/foods.csv.zip
wget http://phenol-explorer.eu/system/downloads/current/publications.csv.zip

unzip composition-data.xlsx.zip
unzip compounds.csv.zip
unzip foods.csv.zip
unzip publications.csv.zip

rm composition-data.xlsx.zip
rm compounds.csv.zip
rm foods.csv.zip
rm publications.csv.zip