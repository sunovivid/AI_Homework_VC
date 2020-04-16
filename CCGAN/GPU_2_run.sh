#!/bin/bash

echo GPU NUMBER 2;
CUDA_VISIBLE_DEVICES=2 python3 -u train_hpchang.py || exit 1;


