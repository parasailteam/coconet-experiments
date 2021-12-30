
# clean process
cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp && parallel-ssh -i -t 0 -p 200 -h pssh.host "pkill -9 python3" 

# original megatron
cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp &&  parallel-ssh -i -t 0 -p 200 -h pssh.host "sudo cp /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx/examples/test.xml ~/test.xml && cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp && sudo cp /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx/examples/test_32.xml ~/test.xml && cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx && GPUS_PER_NODE=16 NNODES=4  MASTER_ADDR=10.217.90.55 MASTER_PORT=12345 LOG_NAME=origin_175B_4VM_mbs12_gbs100 MP_BARRIER=1   WARMUP_BATCH=10 NCCL_DEBUG=VERSION LD_PRELOAD=/msrhyper-ddn/hai8/saemal/nccl-2.8.4-1/build/lib/libnccl.so  sh ./p2pfusion_experiments/time_break_down_experiments/time_break_down_4VM_12bs.sh" 2>&1 | tee /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx/p2pfusion_experiments/time_break_down_logs/origin_175B_4VM_mbs12_gbs100.log

cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp && parallel-ssh -i -t 0 -p 200 -h pssh.host "pkill -9 python3" 
# p2p megatronv2
cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp &&  parallel-ssh -i -t 0 -p 200 -h pssh.host "sudo cp /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx/examples/test.xml ~/test.xml && cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp && sudo cp /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx/examples/test_32.xml ~/test.xml && cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx && GPUS_PER_NODE=16 NNODES=4  MASTER_ADDR=10.217.90.55 MASTER_PORT=12345 LOG_NAME=p2pmeg_175B_4VM_mbs12_gbs100 MP_BARRIER=1   P2P_MEG=1 WARMUP_BATCH=10 NCCL_DEBUG=VERSION  LD_PRELOAD=/msrhyper-ddn/hai8/saemal/nccl-2.8.4-1/build/lib/libnccl.so sh ./p2pfusion_experiments/time_break_down_experiments/time_break_down_4VM_12bs.sh" 2>&1 | tee /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx/p2pfusion_experiments/time_break_down_logs/p2pmeg_175B_4VM_mbs12_gbs100.log

cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp && parallel-ssh -i -t 0 -p 200 -h pssh.host "pkill -9 python3" 
# p2p shard
cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp &&  parallel-ssh -i -t 0 -p 200 -h pssh.host "sudo cp /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx/examples/test.xml ~/test.xml && cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp && sudo cp /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx/examples/test_32.xml ~/test.xml && cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx && GPUS_PER_NODE=16 NNODES=4  MASTER_ADDR=10.217.90.55 MASTER_PORT=12345 LOG_NAME=p2pmeg_175B_4VM_mbs12_gbs100 MP_BARRIER=1  P2P_SHARD=1 WARMUP_BATCH=10 NCCL_DEBUG=VERSION LD_PRELOAD=/msrhyper-ddn/hai8/saemal/nccl-2.8.4-1/build/lib/libnccl.so  sh ./p2pfusion_experiments/time_break_down_experiments/time_break_down_4VM_12bs.sh" 2>&1 | tee /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx/p2pfusion_experiments/time_break_down_logs/p2pshard_175B_4VM_mbs12_gbs100.log

cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp && parallel-ssh -i -t 0 -p 200 -h pssh.host "pkill -9 python3" 
cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp &&  parallel-ssh -i -t 0 -p 200 -h pssh.host "sudo cp /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx/examples/test.xml ~/test.xml && cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp && sudo cp /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx/examples/test_32.xml ~/test.xml && cd /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx && GPUS_PER_NODE=16 NNODES=4  MASTER_ADDR=10.217.90.55 MASTER_PORT=12345 LOG_NAME=p2pfusion_175B_4VM_mbs12_gbs100 MP_BARRIER=1  PIPELINED_P2P_FUSION=1 WARMUP_BATCH=10 NCCL_DEBUG=VERSION  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple sh ./p2pfusion_experiments/time_break_down_experiments/time_break_down_4VM_12bs.sh" 2>&1 | tee /msrhyper-ddn/hai8/v-junhuang/coconet_exp/repos/Megatron-LMx/p2pfusion_experiments/time_break_down_logs/p2pfusion_175B_4VM_mbs12_gbs100.log