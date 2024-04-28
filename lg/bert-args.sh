#!/bin/sh

echo --num-layers 24
echo --hidden-size 1024
echo --num-attention-heads 16
echo --seq-length 512
echo --max-position-embeddings 512
echo --micro-batch-size 4
echo --global-batch-size 8
echo --lr 0.0001
echo --train-iters 2000000
echo --lr-decay-iters 990000
echo --lr-decay-style linear
echo --min-lr 0.00001
echo --weight-decay 1e-2
echo --lr-warmup-fraction .01
echo --clip-grad 1.0
echo --fp16
