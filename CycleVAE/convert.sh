#!/bin/bash
python3 -u convert.py --model_type=VAE3 --model_path=model/VAE3_CC --convert_path=result/VAE3_cycle || exit 1;
