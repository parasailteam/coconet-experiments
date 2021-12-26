#!/bin/bash

GPUS_PER_NODE=`echo $NPROC`
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
BATCH_SIZE=8
DATA_PATH="$1"/my-bert_text_sentence #/philly/rr3/msrhyperprojvc2_scratch/saemal/amir/data/wikidata/raid/Megatron-LM/my-bert_text_sentence
CHECKPOINT_PATH=/mnt/checkpoint #/philly/rr3/msrhyperprojvc2_scratch/saemal/amir/data/wikidata/raid/checkpoint

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
#export NCCL_PROTO=Simple
#export NCCL_DEBUG=INFO
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --model-parallel-size $GPUS_PER_NODE \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size $BATCH_SIZE \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 100 \
       --data-path $DATA_PATH \
       --vocab-file "$1"/bert-large-uncased-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --log-interval 1 \
       --fp16
