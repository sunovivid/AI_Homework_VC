#!/bin/bash

echo GPU NUMBER 0;
CUDA_VISIBLE_DEVICES=0 python3 -u train_hpchang.py || exit 1;


