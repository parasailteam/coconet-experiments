parallel-ssh -i -t 0 -p 32 -h pssh.host "sudo docker stop jun_superbench  && sudo docker rm jun_superbench"

parallel-ssh -i -t 0 -p 32 -h pssh.host "sudo docker rmi  a97933db90b1"

parallel-ssh -i -t 0 -p 32 -h pssh.host "sudo docker stop superbench  && sudo docker rm superbench"

parallel-ssh -i -t 0 -h pssh.host "sudo docker run -it -d --name=sccl_dev --privileged --net=host --ipc=host --gpus=all -v $HOME/superbench-results:/superbench/Results -v /mnt:/mnt -v /mnt/tmp:/tmp -v /opt:/opt2 bench.azurecr.io/superbench:cuda11.1.1-20210204 bash"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cd /mnt && python3 -m pip uninstall -y torch'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cd /mnt && rm librarys -r && rm repos -r && rm apex -r'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cd /mnt && python3 -m pip uninstall -y apex'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cd /mnt && mkdir librarys && cd librarys &&  git clone https://v-junhuang:pdxhxdy52ap5yijqahsxemjdgsvvhutbgz67fsnilzj6oxg6awwq@msrasrg.visualstudio.com/PyTorch-x/_git/PyTorch-x && mv PyTorch-x pytorch && cd pytorch && git checkout v1.8.1_sckl &&  git submodule sync &&  git submodule update --init --recursive &&  cd third_party/nccl && rm nccl -r &&   git clone https://v-junhuang:qmrfjvdyzfgdvzspxg2pnrrlldlyf6f5sapuary5boi5xrraeiya@msrasrg.visualstudio.com/nccl-x/_git/nccl-x && mv nccl-x nccl && cd nccl && git checkout custom_func  && cd /home/saemal/librarys/pytorch && python3 setup.py install'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cd /mnt && mkdir repos && cd repos &&  git clone https://v-junhuang:4pbrowdthzomsen6sabplnnnbyma6pyvpg4vkb7un2vszenamiwa@msrasrg.visualstudio.com/Megatron-LMx/_git/Megatron-LMx && cd ~/repos/Megatron-LMx && git checkout pipeline-fusion-experiment && git pull'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cd /mnt &&  python3 -m pip uninstall -y apex  && git clone https://github.com/NVIDIA/apex  && cd apex && CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include pip install -v --disable-pip-version-check --no-cache-dir --global-option='--cpp_ext' --global-option='--cuda_ext' ./'"


parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cd /mnt && python3 -m pip uninstall -y torch && pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cd /mnt && python3 -m pip uninstall -y torch'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cd /mnt/librarys/pytorch && python3 setup.py install'"

#xml copy

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cp /mnt/repos/Megatron-LMx/examples/test.xml /root'"
-------------------------------

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cd /mnt/repos/Megatron-LMx && git pull'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'pkill -9 python3'"

# 2VM origin 
parallel-ssh -i -t 0 -p 200 -h pssh2.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_13B_2VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=11  sh ./examples/pretrain_gpt3_13B_2VMs.sh"' 2>&1 | tee ./origin_13B_2VM_bs100.log

#2VM AR
parallel-ssh -i -t 0 -p 200 -h pssh2.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_13B_2VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=11  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_2VMs.sh"' 2>&1 | tee ./ar_13B_2VM_bs100.log

#2VM RS
parallel-ssh -i -t 0 -p 200 -h pssh2.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_13B_2VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=11   sh ./examples/pretrain_gpt3_13B_2VMs.sh"' 2>&1 | tee ./rs_13B_2VM_bs100.log


#2VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh2.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_13B_2VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=11  sh ./examples/pretrain_gpt3_13B_2VMs.sh"' 2>&1 | tee ./pipe_13B_2VM_bs100.log

-------

#4VM


# 4VM origin 
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pretrain_gpt3_175B_4VM_16layer_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=11 NCCL_DEBUG=VERSION   sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs1.sh"' 2>&1 | tee ./pretrain_gpt3_175B_4VM_16layer_bs1.log

#4VM AR
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_13B_4VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=11  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_4VMs.sh"' 2>&1 | tee ./ar_13B_4VM_bs100.log

#4VM RS
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_13B_4VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=11   sh ./examples/pretrain_gpt3_13B_4VMs.sh"' 2>&1 | tee ./rs_13B_4VM_bs100.log


#4VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_13B_4VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=11  sh ./examples/pretrain_gpt3_13B_4VMs.sh"' 2>&1 | tee ./pipe_13B_4VM_bs100.log

--------

#8 VM
# 8VM origin 
parallel-ssh -i -t 0 -p 200 -h pssh8.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=8 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_13B_8VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=11 NCCL_DEBUG=VERSION   sh ./examples/pretrain_gpt3_13B_8VMs.sh"' 2>&1 | tee ./origin_13B_8VM_bs100.log

#8VM AR
parallel-ssh -i -t 0 -p 200 -h pssh8.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=8 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_13B_8VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=11  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_8VMs.sh"' 2>&1 | tee ./ar_13B_8VM_bs100.log

#8VM RS
parallel-ssh -i -t 0 -p 200 -h pssh8.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=8 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_13B_8VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=11   sh ./examples/pretrain_gpt3_13B_8VMs.sh"' 2>&1 | tee ./rs_13B_8VM_bs100.log


#8VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh8.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=8 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_13B_8VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=11  sh ./examples/pretrain_gpt3_13B_4VMs.sh"' 2>&1 | tee ./pipe_13B_8VM_bs100.log



--------

source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_13B_2VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=11  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_2VMs.sh

source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_13B_2VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=11  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_2VMs.sh


source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_13B_2VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple  NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=11 NCCL_BUFFSIZE=1572864 sh ./examples/pretrain_gpt3_13B_2VMs.sh

source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_13B_2VM_bs100 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple  NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=11  NCCL_BUFFSIZE=1572864  sh ./examples/pretrain_gpt3_13B_2VMs.sh



parallel-ssh -i -t 0 -p 200 -h pssh2.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_13B_2VM_10bs MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=11  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_2VMs_10bs.sh"' 2>&1 | tee ./ar_13B_2VM_10bs.log


source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_13B_2VM_10bs MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=3  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_2VMs_10bs.sh


source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_13B_2VM_10bs MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=3  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_2VMs_10bs.sh


source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_13B_2VM_10bs MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=11  sh ./examples/pretrain_gpt3_13B_2VMs_10bs.sh


source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_13B_2VM_10bs MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=11  sh ./examples/pretrain_gpt3_13B_2VMs_10bs.sh





# 2VM origin 
parallel-ssh -i -t 0 -p 200 -h pssh2.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_13B_2VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=3 NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_2VMs.sh"' 2>&1 | tee ./origin_13B_2VM_bs8.log

#2VM AR
parallel-ssh -i -t 0 -p 200 -h pssh2.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_13B_2VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=3  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_2VMs.sh"' 2>&1 | tee ./ar_13B_2VM_bs8.log

#2VM RS
parallel-ssh -i -t 0 -p 200 -h pssh2.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_13B_2VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=3 NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_2VMs.sh"' 2>&1 | tee ./rs_13B_2VM_bs8.log


#2VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh2.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_13B_2VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=3  sh ./examples/pretrain_gpt3_13B_2VMs.sh"' 2>&1 | tee ./pipe_13B_2VM_bs8.log

-------

#4VM


# 4VM origin 
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_13B_4VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=5 NCCL_DEBUG=VERSION   sh ./examples/pretrain_gpt3_13B_4VMs.sh"' 2>&1 | tee ./origin_13B_4VM_bs8.log

#4VM AR
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_13B_4VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=5  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_4VMs.sh"' 2>&1 | tee ./ar_13B_4VM_bs8.log

#4VM RS
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_13B_4VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=5   sh ./examples/pretrain_gpt3_13B_4VMs.sh"' 2>&1 | tee ./rs_13B_4VM_bs8.log


#4VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_13B_4VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=5  sh ./examples/pretrain_gpt3_13B_4VMs.sh"' 2>&1 | tee ./pipe_13B_4VM_bs8.log

--------

#8 VM
# 8VM origin 
parallel-ssh -i -t 0 -p 200 -h pssh8.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=8 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_13B_8VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=9 NCCL_DEBUG=VERSION   sh ./examples/pretrain_gpt3_13B_8VMs.sh"' 2>&1 | tee ./origin_13B_8VM_bs8.log

#8VM AR
parallel-ssh -i -t 0 -p 200 -h pssh8.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=8 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_13B_8VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=9  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_8VMs.sh"' 2>&1 | tee ./ar_13B_8VM_bs8.log

#8VM RS
parallel-ssh -i -t 0 -p 200 -h pssh8.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=8 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_13B_8VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=9   sh ./examples/pretrain_gpt3_13B_8VMs.sh"' 2>&1 | tee ./rs_13B_8VM_bs8.log


#8VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh8.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=8 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_13B_8VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=9  sh ./examples/pretrain_gpt3_13B_8VMs.sh"' 2>&1 | tee ./pipe_13B_8VM_bs8.log


--------
# 16 VM
# 16VM origin 
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_13B_16VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=17 NCCL_DEBUG=VERSION   sh ./examples/pretrain_gpt3_13B_16VMs.sh"' 2>&1 | tee ./origin_13B_16VM_bs8.log

#16VM AR
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_13B_16VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=17  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_13B_16VMs.sh"' 2>&1 | tee ./ar_13B_16VM_bs8.log

#16VM RS
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_13B_16VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=17   sh ./examples/pretrain_gpt3_13B_16VMs.sh"' 2>&1 | tee ./rs_13B_16VM_bs8.log


#16VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_13B_16VM_bs8 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=17  sh ./examples/pretrain_gpt3_13B_16VMs.sh"' 2>&1 | tee ./pipe_13B_16VM_bs8.log



#16VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_175B_16VM_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=17  sh ./examples/pretrain_gpt3_175B_16VM.sh"' 2>&1 | tee ./pipe_175B_16VM_bs1.log



#16VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_175B_16VM_16layers_bs6 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=17  sh ./examples/pretrain_gpt3_175B_16VM_16layer.sh"' 2>&1 | tee ./pipe_175B_16VM_16layers_bs6.log


------------

# 16VM origin 
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_175B_16VM_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=17 NCCL_DEBUG=VERSION   sh ./examples/pretrain_gpt3_175B_16VM.sh"' 2>&1 | tee ./origin_175B_16VM_bs1.log

#16VM AR
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_175B_16VM_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=17  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_175B_16VM.sh"' 2>&1 | tee ./ar_175B_16VM_bs1.log

#16VM RS
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_175B_16VM_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=17   sh ./examples/pretrain_gpt3_175B_16VM.sh"' 2>&1 | tee ./rs_175B_16VM_bs1.log




#16VM pipe

parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_175B_16VM_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=17  sh ./examples/pretrain_gpt3_175B_16VM.sh"' 2>&1 | tee ./pipe_175B_16VM_bs1.log

-------------

# 16VM origin 
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_175B_16VM_96layers_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=17 NCCL_DEBUG=VERSION   sh ./examples/pretrain_gpt3_175B_16VM_96layer.sh"' 2>&1 | tee ./origin_175B_16VM_96layers_bs1.log

#16VM AR
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_175B_16VM_96layers_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=17  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_175B_16VM_96layer.sh"' 2>&1 | tee ./ar_175B_16VM_96layers_bs1.log

#16VM RS
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_175B_16VM_96layer_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=17   sh ./examples/pretrain_gpt3_175B_16VM_96layer.sh"' 2>&1 | tee ./rs_175B_16VM_96layers_bs1.log




#16VM pipe

parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_175B_16VM_96layers_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=17  sh ./examples/pretrain_gpt3_175B_16VM_96layer.sh"' 2>&1 | tee ./pipe_175B_16VM_96layers_bs1.log

---------

#175 16layer bs6
# 16VM origin 
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_175B_16VM_bs6 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=17 NCCL_DEBUG=VERSION   sh ./examples/pretrain_gpt3_175B_16VM.sh"' 2>&1 | tee ./origin_175B_16VM_bs6.log

#16VM AR
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_175B_16VM_bs6 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=17  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_175B_16VM.sh"' 2>&1 | tee ./ar_175B_16VM_bs6.log

#16VM RS
parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_175B_16VM_bs6_breakdown MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=17   sh ./examples/pretrain_gpt3_175B_16VM.sh"' 2>&1 | tee ./rs_175B_16VM_bs6_breakdown.log




#16VM pipe

parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=15 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_175B_16VM_bs6 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=17  sh ./examples/pretrain_gpt3_175B_16VM.sh"' 2>&1 | tee ./pipe_175B_16VM_bs6.log



export PATH=/home/saemal/.local/bin:$PATH





---------------

# v100  

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cd /mnt/repos/Megatron-LMx && git pull'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'pkill -9 python3'"

LD_PRELOAD=/msrhyper-weka/saemal/nccl_org/build/lib/libnccl.so        LD_LIBRARY_PATH=/msrhyper-weka/saemal/nccl_org/build/lib/

# pipe 4 VM bs1 layer16
parallel-ssh -i -t 0 -p 200 -h pssh.host 'bash -c  "cd ~/repos/Megatron-LMx && GPUS_PER_NODE=16 NNODES=4  MASTER_ADDR=10.184.185.44 MASTER_PORT=12345   LD_PRELOAD=/home/saemal/librarys/pytorch/build/nccl/lib/libnccl.so  LD_LIBRARY_PATH=/home/saemal/librarys/pytorch/build/nccl/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_175B_4VM_bs1 MP_BARRIER=1     NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=98  sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs1.sh"' 2>&1 | tee ./pipe_175B_4VM_bs1.log

bash -c "GPUS_PER_NODE=16 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.184.185.44 MASTER_PORT=12345   LD_PRELOAD=/home/saemal/librarys/pytorch/build/nccl/lib/libnccl.so  LD_LIBRARY_PATH=/home/saemal/librarys/pytorch/build/nccl/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_175B_4VM_bs1 MP_BARRIER=1     NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=98  sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs1.sh"

bash -c "GPUS_PER_NODE=16 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.184.185.44 MASTER_PORT=12345   LD_PRELOAD=/home/saemal/librarys/pytorch/build/nccl/lib/libnccl.so  LD_LIBRARY_PATH=/home/saemal/librarys/pytorch/build/nccl/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_175B_4VM_bs1 MP_BARRIER=1     NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=98  sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs1.sh"

bash -c "GPUS_PER_NODE=16 NNODES=4 NODE_RANK=2 MASTER_ADDR=10.184.185.44 MASTER_PORT=12345   LD_PRELOAD=/home/saemal/librarys/pytorch/build/nccl/lib/libnccl.so  LD_LIBRARY_PATH=/home/saemal/librarys/pytorch/build/nccl/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_175B_4VM_bs1 MP_BARRIER=1     NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=98  sh ~/repos/Megatron-LMx/examples/pretrain_gpt3_175B_4VM_16layer_bs1.sh"

bash -c "GPUS_PER_NODE=16 NNODES=4 NODE_RANK=3 MASTER_ADDR=10.184.185.44 MASTER_PORT=12345   LD_PRELOAD=/home/saemal/librarys/pytorch/build/nccl/lib/libnccl.so  LD_LIBRARY_PATH=/home/saemal/librarys/pytorch/build/nccl/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_175B_4VM_bs1 MP_BARRIER=1     NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=98  sh ~/repos/Megatron-LMx/examples/pretrain_gpt3_175B_4VM_16layer_bs1.sh"




parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'cd /mnt/repos/Megatron-LMx && git pull'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i sccl_dev bash -c 'pkill -9 python3'"


parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pretrain_gpt3_175B_4VM_16layer_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=102 NCCL_DEBUG=VERSION   sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs1.sh"' 2>&1 | tee ./pretrain_gpt3_175B_4VM_16layer_bs1.log

#4VM AR
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_pretrain_gpt3_175B_4VM_16layer_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=102  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs1.sh"' 2>&1 | tee ./ar_pretrain_gpt3_175B_4VM_16layer_bs1.log

#4VM RS
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_pretrain_gpt3_175B_4VM_16layer_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=102   sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs1.sh"' 2>&1 | tee ./rs_pretrain_gpt3_175B_4VM_16layer_bs1.log


#4VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_pretrain_gpt3_175B_4VM_16layer_bs1 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=102  sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs1.sh"' 2>&1 | tee ./pipe_pretrain_gpt3_175B_4VM_16layer_bs1.log



--------
#bs 2

parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pretrain_gpt3_175B_4VM_16layer_bs2 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=52 NCCL_DEBUG=VERSION   sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs2.sh"' 2>&1 | tee ./pretrain_gpt3_175B_4VM_16layer_bs2.log

#4VM AR
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_pretrain_gpt3_175B_4VM_16layer_bs2 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=102  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs2.sh"' 2>&1 | tee ./ar_pretrain_gpt3_175B_4VM_16layer_bs2.log

#4VM RS
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_pretrain_gpt3_175B_4VM_16layer_bs2 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=102   sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs2.sh"' 2>&1 | tee ./rs_pretrain_gpt3_175B_4VM_16layer_bs2.log


#4VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_pretrain_gpt3_175B_4VM_16layer_bs2 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=102  sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs2.sh"' 2>&1 | tee ./pipe_pretrain_gpt3_175B_4VM_16layer_bs2.log


--------
#bs 4

parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pretrain_gpt3_175B_4VM_16layer_bs4 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=52 NCCL_DEBUG=VERSION   sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs4.sh"' 2>&1 | tee ./pretrain_gpt3_175B_4VM_16layer_bs4.log

#4VM AR
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_pretrain_gpt3_175B_4VM_16layer_bs4 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=102  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs4.sh"' 2>&1 | tee ./ar_pretrain_gpt3_175B_4VM_16layer_bs4.log

#4VM RS
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_pretrain_gpt3_175B_4VM_16layer_bs4 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=102   sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs4.sh"' 2>&1 | tee ./rs_pretrain_gpt3_175B_4VM_16layer_bs4.log


#4VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_pretrain_gpt3_175B_4VM_16layer_bs4 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=102  sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs4.sh"' 2>&1 | tee ./pipe_pretrain_gpt3_175B_4VM_16layer_bs4.log


--------
#bs 6

parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pretrain_gpt3_175B_4VM_16layer_bs6 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include WARMUP_BATCH=52 NCCL_DEBUG=VERSION   sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs6.sh"' 2>&1 | tee ./pretrain_gpt3_175B_4VM_16layer_bs6.log

#4VM AR
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=ar_pretrain_gpt3_175B_4VM_16layer_bs6 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_MEG=1  WARMUP_BATCH=102  NCCL_DEBUG=VERSION  sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs6.sh"' 2>&1 | tee ./ar_pretrain_gpt3_175B_4VM_16layer_bs6.log

#4VM RS
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=rs_pretrain_gpt3_175B_4VM_16layer_bs6 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include  P2P_SHARD=1   WARMUP_BATCH=102   sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs6.sh"' 2>&1 | tee ./rs_pretrain_gpt3_175B_4VM_16layer_bs6.log


#4VM PIPE  channels buffsize
parallel-ssh -i -t 0 -p 200 -h pssh4.host 'sudo docker exec -i sccl_dev bash -c "source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=pipe_pretrain_gpt3_175B_4VM_16layer_bs6 MP_BARRIER=1  CPATH=/opt/conda/lib/python3.8/site-packages/pybind11/include   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml NCCL_PROTO=Simple NCCL_DEBUG=VERSION  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=102  sh ./examples/pretrain_gpt3_175B_4VM_16layer_bs6.sh"' 2>&1 | tee ./pipe_pretrain_gpt3_175B_4VM_16layer_bs6.log