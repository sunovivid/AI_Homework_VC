#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 -u test_evaluation.py --model_type=MD --CC=1 || exit 1;
