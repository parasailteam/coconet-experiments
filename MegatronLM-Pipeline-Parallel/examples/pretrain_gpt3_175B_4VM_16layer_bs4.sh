#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=./webtext2/my-gpt2_text_document
#CHECKPOINT_PATH=<Specify path>
#export NODE_RANK=$(sh ~/rank.sh) 
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
	--tensor-model-parallel-size 8 \
	--pipeline-model-parallel-size 4 \
        --num-layers 2 \
        --partition 0 1 1 0\
        --hidden-size 12288 \
        --num-attention-heads 96 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
	--micro-batch-size 4 \
	--global-batch-size 104 \
       --train-iters 30 \
       --lr-decay-iters 320000 \
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
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 1 \
       --fp16 \
       --DDP-impl local \
       --checkpoint-activations \
       --no-scaled-masked-softmax-fusion 
       #--checkpoint-activations 
