#!/bin/bash
set -x
#export NCCL_COMM_ID="localhost:12101"
GPUS_PER_NODE=16
# Change for multinode config
MASTER_ADDR=10.184.185.21
MASTER_PORT=10605
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/philly/rr3/msrhyperprojvc2_scratch/saemal/amir/data/wikidata/raid/Megatron-LM/my-bert_text_sentence
CHECKPOINT_PATH=/philly/rr3/msrhyperprojvc2_scratch/saemal/amir/data/wikidata/raid/checkpoint

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
#-hostfile hostfile
#which python
#python -m torch.distributed.launch $DISTRIBUTED_ARGS \
b=1
while [ "$b" -le 512 ] 
do
       echo
       echo $b
       echo
       mpirun  --allow-run-as-root  -hostfile hostfile python  \
              pretrain_bert.py \
              --model-parallel-size 32 \
              --num-layers 24 \
              --hidden-size 1024 \
              --num-attention-heads 32 \
              --batch-size $b \
              --seq-length 512 \
              --max-position-embeddings 512 \
              --train-iters 100 \
              --data-path $DATA_PATH \
              --vocab-file /philly/rr3/msrhyperprojvc2_scratch/saemal/amir/data/wikidata/raid/Megatron-LM/bert-large-uncased-vocab.txt \
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
       
       b=`expr $b '*' 2`
done || exit 1
