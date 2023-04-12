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


# for lr in 5e-6 1e-5 5e-5
# for epochs in 50 100
# for max_seq_length in 64 128

for lr in 1e-5
do
    for epochs in 5
    do
        for max_seq_length in 64
        do
            echo 'LR: ' $lr
            echo 'Epoch: ' $epochs
            echo 'Maximum sequence length: ' $max_seq_length

            python run_kgbert_link_prediction.py \
                --task_name fakg \
                --do_train \
                --do_eval \
                --data_dir ../../../outputs/kgc/data/annotations \
                --bert_model ../../../models/biobert_v1.1 \
                --max_seq_length $max_seq_length \
                --train_batch_size 32 \
                --learning_rate $lr \
                --num_train_epochs $epochs \
                --output_dir "../../../outputs/kgc/kg-bert/annotations/hpo/lr_${lr}_epoch_${epochs}_maxseq_${max_seq_length}" \
                --gradient_accumulation_steps 1 \
                --eval_batch_size 128 \
                --num_eval 50

            echo ''
            echo ''
            echo ''
        done
    done
done
