#!/bin/bash

srun --time=05:00:00 --ntasks 1 --cpus-per-task 4 --gres gpu:1 --mem 8G --pty bash
