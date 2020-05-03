#!/bin/bash
mkdir -p result

python3 preprocess-train.py
python3 preprocess-eval.py
