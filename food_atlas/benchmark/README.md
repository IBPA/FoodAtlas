# FoodAtlas Benchmark

This repository compares FoodAtlas to FoodMine which contains abundant chemical information for cocoa and garlic.

## Prerequisites

You want to make sure that FoodMine repository is cloned in this directory and that the FooDB is downloaded in `FoodAtlas/data` directory.
- Download FooDB by following [`FoodAtlas/data/README.md`](../../data/README.md).
- Download FoodMine by following the below:
```console
# Clone FoodMine in this repository.
git clone git@github.com:fhooton/FoodMine.git
```

## How to Run

### FoodMine Benchmark

- The following command generates `venn_cocoa.svg` and `venn_garlic.svg` in `FoodAtlas/outputs/benchmark` directory:
```console
cd ../..
python -m food_atlas.benchmark.plot_venn_diagram_foodmine
cd food_atlas/benchmark
```

### FooDB Benchmark

- (Optional) The downloaded FooDB does not have PubChem CIDs which are required for the benchmark. The following command dumps `FoodAtlas/outputs/benchmark/inchikeys_foodb.txt`, which can be used to retrieve corresponding PubChem CIDs [here](https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi). Once you retrieved and unzipped the file, put it in the same directory and rename it to `inchikeys_to_cids_foodb.txt`.
```console
cd ../..
python -m food_atlas.benchmark.dump_inchikeys_foodb
cd food_atlas/benchmark
```

- The following command generates `venn_fa_foodb_chemicals.svg`, `venn_fa_foodb_foods.svg`, and `venn_triplets.svg` in `FoodAtlas/outputs/benchmark` directory:
```console
cd ../..
python -m food_atlas.benchmark.plot_venn_diagram_foodb
cd food_atlas/benchmark
```

### Statistics of Indexed Journal Articles in Databases

- First, download the indexed journal databases (AGRICOLA, CABI, Web of Science, Scopus) by following the below:
```console
cd indexed_journals
./download_data.sh
cd ..
```

- Run the script.
```console
python -m food_atlas.benchmark.get_ext_db_stats
```
