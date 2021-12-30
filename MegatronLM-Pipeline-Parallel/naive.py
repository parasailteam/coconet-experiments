import os
import time
import datetime
import torch
import torch.nn.functional as F
import torch.nn as nn
torch.backends.cudnn.benchmark = True

### example command to run on single node
# T='allreduce' python3 -m torch.distributed.launch --nproc_per_node=4 --use_env naive.py
# T='allgather' python3 -m torch.distributed.launch --nproc_per_node=4 --use_env naive.py

### example command to run on multiple nodes
# on NODE 0: export T='allgather' &&  python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="10.0.0.6" --master_port=12345 --use_env /mnt/torch_matmul_shard.py
# on NODE 1: export T='allgather' &&  python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="10.0.0.6" --master_port=12345 --use_env /mnt/torch_matmul_shard.py
'''
torch.distributed.init_process_group(backend='nccl')
'''
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

C = world_size
M = int(os.environ.get('M', '12288'))
K = int(os.environ.get('K', '12288'))
N = int(os.environ.get('N', '16000'))
if local_rank == 0:
    print('\nusing {} GPUs. Workload size M={} K={} N={}'.format(C, M, K, N))

T = os.environ.get('T', 'allreduce')
device = os.environ.get('D', '0')
warm_up = 50000000
test = 50

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

total_cost = []
comp_cost = []
comm_cost = []

if T == 'allreduce':
    x = torch.ones(M, K // C).cuda()
    y = torch.ones(K // C, N).cuda()
    for i in range(warm_up):
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        torch.distributed.all_reduce(z, op=torch.distributed.ReduceOp.SUM)
        torch.cuda.synchronize()

    for i in range(test):
        start = time.perf_counter()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        
        comm_start = time.perf_counter()
        if C != 1:
            torch.distributed.all_reduce(z, op=torch.distributed.ReduceOp.SUM)
            torch.cuda.synchronize()

        end = time.perf_counter()
        comp_cost.append((comm_start - start) * 1000)
        comm_cost.append((end - comm_start) * 1000)
        total_cost.append((end - start) * 1000)

    if local_rank == 0:
        print('allreduce | GPU: {} | comp: {:.3f} ms | comm: {:.3f} ms | e2e: {:.3f} ms'.format(
            world_size, sum(comp_cost) / len(comp_cost), sum(comm_cost) / len(comm_cost), sum(total_cost) / len(total_cost)))

elif T == 'allgather':
    x = torch.ones(M // C, K).cuda().half()
    y = torch.ones(K, N).cuda().half()

    tensor_list = []
    for i in range(world_size):
        tensor_list.append(torch.zeros(M // C, N).cuda())

    for i in range(warm_up):
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        torch.distributed.all_gather(tensor_list, z)
        torch.cuda.synchronize()

    for i in range(test):
        start = time.perf_counter()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()

        comm_start = time.perf_counter()
        if C != 1:
            torch.distributed.all_gather(tensor_list, z)
            # z = torch.cat(tensor_list, 0)
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        comp_cost.append((comm_start - start) * 1000)
        comm_cost.append((end - comm_start) * 1000)
        total_cost.append((end - start) * 1000)

    if local_rank == 0:
        print('allgather | GPU: {} | comp: {:.3f} ms | comm: {:.3f} ms | e2e: {:.3f} ms'.format(
            world_size, sum(comp_cost) / len(comp_cost), sum(comm_cost) / len(comm_cost), sum(total_cost) / len(total_cost)))
elif T == 'batchmm':
    x_in = torch.zeros(M, K, K).half().cuda()
    x_0 = torch.ones(M, K, N).half().cuda()
    x_1 = torch.ones(M, N, K).half().cuda()
    start = time.perf_counter()
    for i in range(warm_up):
        #z = F.linear(x, y)
        #print(z)
        z = torch.bmm(x_0, x_1)
        torch.cuda.synchronize()
        if time.perf_counter() - start > 10:
            break
    for i in range(test):
        #start = time.perf_counter()
        start_event.record()
        #z = F.linear(x, y)
        z = torch.bmm( x_0, x_1)         
        end_event.record()

        torch.cuda.synchronize()
        #end = time.perf_counter()
        #total_cost.append((end - start) * 1000)
        total_cost.append(start_event.elapsed_time(end_event))
    if local_rank == 0:
        print('single GPU | e2e: {} ms'.format(sum(total_cost) / len(total_cost)))
    
elif T =='single':

    x = torch.ones(M, K).half().to(f"cuda:{device}")
    print(f"device is :{x.device}")
    #y = torch.ones(N, K).half().cuda()
    y = torch.ones(K, N).half().to(f"cuda:{device}")
    #b = torch.ones(M, N).half().cuda()
    start = time.perf_counter()
    for i in range(warm_up):
        #z = F.linear(x, y)
        #print(z)
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        if time.perf_counter() - start > 10:
            break

    for i in range(test):
        #start = time.perf_counter()
        start_event.record()
        #z = F.linear(x, y)
        z = torch.matmul(x, y)         
        end_event.record()

        torch.cuda.synchronize()
        #end = time.perf_counter()
        #total_cost.append((end - start) * 1000)
        total_cost.append(start_event.elapsed_time(end_event))

    if local_rank == 0:
        print('single GPU | e2e: {} ms'.format(sum(total_cost) / len(total_cost)))
'''
torch.distributed.destroy_process_group()
'''
