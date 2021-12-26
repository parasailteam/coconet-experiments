#! /bin/bash

# Runs the "1.2B" parameter model

GPUS_PER_NODE=`echo $NPROC`
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH="$1"/my-gpt2_text_document
CHECKPOINT_PATH=/mnt/checkpoint

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       --model-parallel-size $GPUS_PER_NODE \
       --num-layers 40 \
       --hidden-size 1536\
       --num-attention-heads 16 \
       --batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 10 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --log-interval 10 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16



set +x
