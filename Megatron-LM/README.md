[Megatron README](OriginalREADME.md)

# Megatron DSL Accelerated

We optimized Megatron-LM by fusing collective operations with multiple pointwise operations. [This DSL](https://github.com/parasailteam/nccl-public/tree/accc-dsl/accc-dsl) has been used for implementing the aforementioned optimization.

The DSL fuses several operations and adds a function to the [NCCL](https://github.com/NVIDIA/nccl). The modified NCCL will be used in PyTorch and the new function will be available in PyTorch. Finally, Megatron-LM uses the function from PyTorch.

This [PyTorch]() branch is ready to use for Megatron-LM-DSL-Acc. It has the modified NCCL in pytorch/third_party/nccl/nccl/ directory. Simply install this PyTorch first and you will be able to use Megatron-LM-DSL-Acc.

To run in a single node, use this script: /examples/pretrain_bert_distributed.sh. Make sure the first 5 lines of pretrain_bert.py (set os environment) are commented.

To run in multiple nodes,use this script: /examples/pretrain_bert_distributed_multinode.sh. Make sure the first 5 lines of pretrain_bert.py (set os environment) are uncommented ans set correctly. Put the node ips in the hostfile.