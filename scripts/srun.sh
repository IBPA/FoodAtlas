#!/bin/bash

srun \
    --time=24:00:00 \
    --ntasks 1 \
    --cpus-per-task 8 \
    --gres gpu:1 \
    --mem 25G \
    --pty bash
