#!/bin/sh
# model: rf/tree/xgb
# fold: 1,2,3,4,5


python3 train.py --model rf --fold 0
python3 train.py --model rf --fold 1
python3 train.py --model rf --fold 2
python3 train.py --model rf --fold 3
python3 train.py --model rf --fold 4
