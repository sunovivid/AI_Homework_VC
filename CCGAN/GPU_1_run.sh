#!/bin/bash

echo GPU NUMBER 1;
CUDA_VISIBLE_DEVICES=1 python3 -u train_hpchang.py || exit 1;


