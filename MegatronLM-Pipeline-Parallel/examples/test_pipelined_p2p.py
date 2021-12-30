import torch
import os
rank = int(os.getenv('RANK','0'))
world_size = int(os.getenv('WORLD_SIZE','1'))
init_method = 'tcp://'
master_ip = os.getenv('MASTER_ADDR', 'localhost')
master_port = os.getenv('MASTER_PORT', '12345')
init_method += master_ip + ':' + master_port
torch.distributed.init_process_group("nccl",world_size=world_size, rank=rank, init_method=init_method)
torch.cuda.set_device(rank % torch.cuda.device_count())
print(torch.distributed.get_world_size())

t = (torch.zeros(2048*1*12288)).type(torch.float16).contiguous().cuda()
t_bias = (torch.zeros(2048*1*12288)).type(torch.float16).contiguous().cuda()

output_list = torch.empty(1).cuda()
#output_list = torch.zeros([2048, 1, 12288]).type(torch.float16).contiguous().cuda()
#torch.distributed.all_reduce(t)
torch.distributed.pipelined_p2p_fusion(output_list,  t, t_bias, 0.5)
torch.cuda.synchronize()

print(f"rank:{torch.cuda.current_device()} output_list:{output_list} input:{t} mean:{t.max()}\n")