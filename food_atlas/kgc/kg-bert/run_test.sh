#!/bin/bash
set -e

# python run_kgbert_link_prediction.py \
#     --task_name kg \
#     --do_train \
#     --do_eval \
#     --data_dir ../../../data/umls \
#     --bert_model bert-base-uncased \
#     --max_seq_length 15 \
#     --train_batch_size 32 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 5.0 \
#     --output_dir ../../../outputs/kgc/kg-bert/umls \
#     --gradient_accumulation_steps 1 \
#     --eval_batch_size 128

python run_kgbert_link_prediction.py \
    --task_name fakg \
    --do_predict \
    --data_dir ../../../outputs/kgc/data/annotations \
    --bert_model ../../../outputs/kgc/kg-bert/annotations/hpo/lr_5e-5_epoch_50_maxseq_128 \
    --output_dir ../../../outputs/kgc/kg-bert/annotations/hpo/lr_5e-5_epoch_50_maxseq_128
