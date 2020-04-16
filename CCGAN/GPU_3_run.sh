#!/bin/bash

echo GPU NUMBER 3;
CUDA_VISIBLE_DEVICES=3 python3 -u train_hpchang.py || exit 1;


