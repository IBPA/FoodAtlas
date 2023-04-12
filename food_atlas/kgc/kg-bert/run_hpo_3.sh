#!/bin/bash
set -e

# for lr in 5e-6 1e-5 5e-5
# for epochs in 50 100
# for max_seq_length in 64 128

for lr in 5e-6 1e-5 5e-5
do
    for epochs in 100
    do
        for max_seq_length in 128
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
                --num_eval 50 \
                --cuda_visible_devices 3

            echo ''
            echo ''
            echo ''
        done
    done
done
