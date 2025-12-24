#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
cd /home/keeg/pi2_training
exec .venv/bin/python train_pi2.py --data all_phases_combined.jsonl --dim 2048 --layers 28 --heads 16 --steps 20000 --batch 1 --lr 1e-4 --output ./output_pi2_1.7B
