#!/bin/bash

GPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
#RANK=$(cat "$file")  
echo $NODE_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DATA_PATH=/msrhyper-ddn/hai8/v-junhuang/coconet_exp/data/megatron-bert-data/my-bert_text_sentence
#CHECKPOINT_PATH=<Specify path>

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
        --tensor-model-parallel-size 16 \
        --pipeline-model-parallel-size 1 \
       --num-layers 5 \
       --hidden-size 3072 \
       --num-attention-heads 32 \
       --micro-batch-size 16 \
       --global-batch-size 256 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 30 \
       --data-path $DATA_PATH \
       --vocab-file /msrhyper-ddn/hai8/v-junhuang/coconet_exp/data/megatron-bert-data/bert-large-uncased-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
#       #--save $CHECKPOINT_PATH \
       #--load $CHECKPOINT_PATH \
#       --tensor-model-parallel-size 4 \
#       --pipeline-model-parallel-size 2 \
#       --no-scaled-masked-softmax-fusion \
#