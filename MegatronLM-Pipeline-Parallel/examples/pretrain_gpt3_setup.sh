# copy a100.key into 000000's ~/.ssh/id_rsa

for i in {0..15}; do printf "vmss16nds-%06X\n" $i >>pssh.host; done
parallel-ssh -i -t 0 -h pssh.host 'echo "AcceptEnv PSSH_NODENUM PSSH_HOST" | sudo tee -a /etc/ssh/sshd_config'
parallel-ssh -i -t 0 -h pssh.host "sudo service ssh restart"

# start nvidia-fabricmanager and nv_peer_mem service

parallel-ssh -i -t 0 -h pssh.host "sudo service nvidia-fabricmanager start && sudo service nv_peer_mem start"
# prepare docker image and container

parallel-ssh -i -t 0 -h pssh.host "sudo docker login bench.azurecr.io -u bench -p 'JbJoJvo=LDuxqHOVeLqNPXJBTqUEKMgq'"

parallel-ssh -i -t 0 -p 32 -h pssh.host "sudo docker pull bench.azurecr.io/superbench:cuda11.1-20210119 && sudo docker rm superbench"

parallel-ssh -i -t 0 -p 32 -h pssh.host "sudo docker pull bench.azurecr.io/superbench:cuda11.1-20210119 && sudo docker rm superbench"

parallel-ssh -i -t 0 -p 32 -h pssh.host "sudo docker rm jun_superbench"

parallel-ssh -i -t 0 -h pssh.host "sudo docker run -it -d --name=jun_superbench --privileged --net=host --ipc=host --gpus=all -v $HOME/superbench-results:/superbench/Results -v /mnt:/mnt -v /mnt/tmp:/tmp -v /opt:/opt2 bench.azurecr.io/superbench:cuda11.1.1-20210204 bash"

parallel-ssh -i -t 0 -h pssh.host "sudo docker run -it -d --name=jun_superbench --privileged --net=host --ipc=host --gpus=all -v $HOME/superbench-results:/superbench/Results -v /mnt:/mnt -v /mnt/tmp:/tmp -v /opt:/opt2 nvcr.io/nvidia/pytorch:20.12-py3 bash"
# prepare .ssh/config

i=0
echo >ssh.config
cat pssh.host | while read -r line; do
    echo -e "Host node$i\n    HostName $line\n    Port 22222\n    IdentityFile /etc/ssh/ssh_host_ed25519_key\n    StrictHostKeyChecking no" >>ssh.config
    i=$((i + 1))
done

parallel-scp -h pssh.host ssh.config ~/

parallel-ssh -i -t 0 -h pssh.host "sudo cp ~/ssh.config /mnt"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i jun_superbench bash -c 'cp /mnt/ssh.config /root/.ssh/config && service ssh start'"

# GPT3 model
i=0
echo >ssh.config
cat pssh_4_and_U.host | while read -r line; do
    echo -e "Host node$i\n    HostName $line\n    Port 22222\n    IdentityFile /etc/ssh/ssh_host_ed25519_key\n    StrictHostKeyChecking no" >>ssh.config
    i=$((i + 1))
done

sudo docker run -it -d --name=jun_superbench --privileged --net=host --ipc=host --gpus=all -v $HOME/superbench-results:/superbench/Results -v /mnt:/mnt -v /mnt/tmp:/tmp -v /opt:/opt2 bench.azurecr.io/superbench:cuda11.1-20210119 bash


parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i jun_superbench bash -c 'cd /mnt && mkdir repos && cd repos &&  git clone https://v-junhuang:4pbrowdthzomsen6sabplnnnbyma6pyvpg4vkb7un2vszenamiwa@msrasrg.visualstudio.com/Megatron-LMx/_git/Megatron-LMx'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i jun_superbench bash -c 'cd /mnt/repos/Megatron-LMx && git checkout pipeline-fusion-experiment && git pull'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i jun_superbench bash -c 'cd /mnt/repos/Megatron-LMx && git pull'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i jun_superbench bash -c ' pkill -9 python'"

parallel-ssh -i -t 0 -h pssh_4VM.host "sudo docker exec -i jun_superbench bash -c ' pkill -9 python3'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i jun_superbench bash -c 'pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html'"

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i jun_superbench bash -c 'cd /mnt && rm apex -r && git clone https://github.com/NVIDIA/apex  && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./'"

parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=16 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.10 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=p2pfusion_fusing P2P_FUSION_FUSING=1 sh ./examples/pretrain_gpt3_175B_16VMs.sh"' 2>&1 | tee ./Results/GPT3_p2p_fusion_fusing.log

parallel-ssh -i -t 0 -p 200 -h pssh16VM.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=16 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.23 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_175B_16VM  sh ./examples/pretrain_gpt3_175B_16VMs.sh"' 2>&1 | tee ./GPT3_origin_175B_16VM.log

parallel-ssh -i -t 0 -p 200 -h pssh_4VM.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.17 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_175B_4VM_p2p_fusion P2P_FUSION=1 sh ./examples/pretrain_gpt3_175B_4VM.sh"' 2>&1 | tee ./GPT3_origin_175B_4VM_p2p_fusion.log

parallel-ssh -i -t 0 -p 200 -h pssh_4VM.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.17 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_175B_layer2_4VM_bs4_warmup4_checkpointing_p2p_fusion P2P_FUSION=1 WARMUP_BATCH=4 MICRO_BATCH=4 sh ./examples/pretrain_gpt3_175B_4VM.sh"' 2>&1 | tee ./GPT3_origin_175B_4VM_layer2_bs4_warmup4_checkpointing_p2p_fusion.log

parallel-ssh -i -t 0 -p 200 -h pssh_4VM.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.17 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_175B_layer2_4VM_bs6_warmup4_checkpointing_p2p_fusion P2P_FUSION=1 WARMUP_BATCH=4 MICRO_BATCH=6 sh ./examples/pretrain_gpt3_175B_4VM.sh"' 2>&1 | tee ./GPT3_origin_175B_4VM_layer2_bs6_warmup4_checkpointing_p2p_fusion.log

parallel-ssh -i -t 0 -p 200 -h pssh_4VM.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.17 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_175B_4VM_bs4_p2p_fusion P2P_FUSION=1 sh ./examples/pretrain_gpt3_175B_4VM_bs4.sh"' 2>&1 | tee ./GPT3_origin_175B_4VM_bs4_p2p_fusion.log

parallel-ssh -i -t 0 -p 200 -h pssh_4VM.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=4 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.17 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=origin_175B_4VM_bs8_p2p_fusion P2P_FUSION=1 sh ./examples/pretrain_gpt3_175B_4VM_bs8.sh"' 2>&1 | tee ./GPT3_origin_175B_4VM_bs8_p2p_fusion.log
# in docker
mpirun --allow-run-as-root -np 160 -H node0:8,node1:8,node2:8,node5:8,node7:8,node8:8,node9:8,node10:8,node11:8,node12:8,node13:8,node14:8,node16:8,node17:8,node18:8,node19:8,node20:8,node21:8,node22:8,node23:8,node24:8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=Ring -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib/libnccl-net.so -x PATH -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml /nccl-tests/build/all_reduce_perf -b 1K -e 8G -f 2 -g 1 -c 0 -n 50

mpirun --allow-run-as-root -np 16 -H node0:8,node1:8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=Ring -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib/libnccl-net.so -x PATH -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml /nccl-tests/build/all_reduce_perf -b 1K -e 8G -f 2 -g 1 -c 0 -n 50

mpirun --allow-run-as-root -np 16 -H node0:8,node1:8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml /nccl-tests/build/all_reduce_perf -b 1K -e 8G -f 2 -g 1 -c 0 -n 50

node19:8,node20:8,node21:8,node22:8,node23:8,node24:8

mpirun --allow-run-as-root -np 168 -H node0:8,node1:8,node2:8,node5:8,node7:8,node8:8,node9:8,node10:8,node11:8,node12:8,node13:8,node14:8,node16:8,node17:8,node18:8,node19:8,node20:8,node21:8,node22:8,node23:8,node24:8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=Ring -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so -x PATH -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml /nccl-tests/build/all_reduce_perf -b 1K -e 8G -f 2 -g 1 -c 0 -n 50

mpirun --allow-run-as-root -np 16 -H node0:8,node1:8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=Ring -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so -x PATH -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml /nccl-tests/build/all_reduce_perf -b 1K -e 8G -f 2 -g 1 -c 0 -n 50

mpirun --allow-run-as-root -np 8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=Ring -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so -x PATH -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml /nccl-tests/build/all_reduce_perf -b 1K -e 8G -f 2 -g 1 -c 0 -n 50

mpirun --allow-run-as-root -np 8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=Ring -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib/libnccl-net.so -x PATH -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml /nccl-tests/build/all_reduce_perf -b 1K -e 8G -f 2 -g 1 -c 0 -n 50

,node6:8,node7:8,node8:8,node9:8,node10:8

node0:8,node1:8,node2:8,node3:8,node4:8
# update pytorch if needed
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

nsys profile --trace=cuda,nvtx -o output_file_gpu sh ./examples/pretrain_gpt_distributed_with_mp.sh

parallel-ssh -i -t 0 -h pssh.host "sudo docker exec -i jun_superbench bash -c 'cp /superbench/Megatron-LMx/log* /mnt'"

scp -i ~/.ssh/a100-SBGPT3-01_key.pem superbench@gpt3test-000000:/mnt/log* .

parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=16 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib:$LD_LIBRARY_PATH LOG_NAME=p2pfusion_fusing P2P_FUSION_FUSING=1   nsys profile --trace=cuda,nvtx -o /mnt/p2pfusion_fusing -y 60000 -d 50000  sh ./examples/pretrain_gpt3_175B_16VMs.sh"' 2>&1 | tee ./Results/GPT3_p2p_fusion_fusing.log

parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=16 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib:$LD_LIBRARY_PATH LOG_NAME=p2pfusion P2P_FUSION=1   nsys profile --trace=cuda,nvtx -o /mnt/p2pfusion -y 60000 -d 50000  sh ./examples/pretrain_gpt3_175B_16VMs.sh"' 2>&1 | tee ./Results/GPT3_p2p_fusion.log

parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=16 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib:$LD_LIBRARY_PATH LOG_NAME=origin    nsys profile --trace=cuda,nvtx -o /mnt/origin -y 60000 -d 50000  sh ./examples/pretrain_gpt3_175B_16VMs.sh"' 2>&1 | tee ./Results/GPT3_origin.log

parallel-ssh -i -t 0 -p 200 -h pssh.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=16 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.10 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.7.8/lib:$LD_LIBRARY_PATH LOG_NAME=origin_barrier  MP_BARRIER=1 sh ./examples/pretrain_gpt3_175B_16VMs.sh"' 2>&1 | tee ./GPT3_origin_barrier.log

parallel-ssh -i -t 0 -p 200 -h pssh18VM.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=18 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.23 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=p2pfusion_175B_18VM_mp_barrier MP_BARRIER=1 P2P_FUSION=1  sh ./examples/pretrain_gpt3_175B_18VMs.sh"' 2>&1 | tee ./GPT3_p2pfusion_175B_18VM_mp_barrier.log



scp superbench@gpt3testlp3-000002:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-000004:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-00000U:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-00000W:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-00000X:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-00000Y:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-00000Z:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-000010:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-000011:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-000012:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-000013:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-000015:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-000016:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-000017:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-000019:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-00001A:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-00001B:/mnt/repos/Megatron-LMx/log* .
scp superbench@gpt3testlp3-00001D:/mnt/repos/Megatron-LMx/log* .

NVIDIA_VISIBLE_DEVICES=4 T=single WORLD_SIZE=1 LOCAL_RANK=0 python3 naive.py
NVIDIA_VISIBLE_DEVICES=5 T=single WORLD_SIZE=1 LOCAL_RANK=0 python3 naive.py
NVIDIA_VISIBLE_DEVICES=6 T=single WORLD_SIZE=1 LOCAL_RANK=0 python3 naive.py
NVIDIA_VISIBLE_DEVICES=7 T=single WORLD_SIZE=1 LOCAL_RANK=0 python3 naive.py

D=3 T=single WORLD_SIZE=1 LOCAL_RANK=0 python3 naive.py
D
=4 T=single WORLD_SIZE=1 LOCAL_RANK=0 python3 naive.py
D=5 T=single WORLD_SIZE=1 LOCAL_RANK=0 python3 naive.py
D=6 T=single WORLD_SIZE=1 LOCAL_RANK=0 python3 naive.py
D=7 T=single WORLD_SIZE=1 LOCAL_RANK=0 python3 naive.py
D=8 T=single WORLD_SIZE=1 LOCAL_RANK=0 python3 naive.py

LD_LIBRARY_PATH=~/nccl-master/build/lib/ mpirun --allow-run-as-root -np 16 -H node0:8,node1:8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=SCKL --tag-output -x NCCL_NET_SHARED_BUFFERS=0 -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=1 -x SCKL_XML_FILE=~/test.xml -x LD_LIBRARY_PATH=~/nccl-master/build/lib/ -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml ~/nccl-tests/build/alltoall_perf  -w 100 -n 100 -b 1024 -e 1048576000 -f 2 -c 1 -g 1 -z 0

mpirun --allow-run-as-root -np 8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=Ring -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so -x PATH -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml /nccl-tests/build/all_reduce_perf -b 1K -e 8G -f 2 -g 1 -c 0 -n 50


LD_LIBRARY_PATH=/opt2/msft/nccl-2.8.3-1/build/lib  mpirun --allow-run-as-root -np 16 -H node0:8,node1:8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=Ring -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so -x PATH -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml /nccl-tests/build/all_reduce_perf -b 1K -e 8G -f 2 -g 1 -c 0 -n 50

LD_LIBRARY_PATH=~/nccl-master/build/lib/ mpirun --allow-run-as-root -np 16 -H node0:8,node1:8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=SCKL --tag-output -x NCCL_NET_SHARED_BUFFERS=0 -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=16 -x SCKL_XML_FILE=~/test.xml   -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:~/nccl-master/build/lib/ -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml ~/nccl-tests/build/alltoall_perf  -w 100 -n 100 -b 1024 -e 1048576000 -f 2 -c 1 -g 1 -z 0


LD_LIBRARY_PATH=~/nccl-master/build/lib/ mpirun --allow-run-as-root -np 8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=SCKL --tag-output -x NCCL_NET_SHARED_BUFFERS=0 -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=16 -x SCKL_XML_FILE=~/test.xml   -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:~/nccl-master/build/lib/ -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml ~/nccl-tests/build/alltoall_perf  -w 100 -n 100 -b 1024 -e 1048576000 -f 2 -c 1 -g 1 -z 0


parallel-ssh -i -t 0 -h pssh_2VM.host "sudo docker exec -i sccl_superbench bash -c 'cp /mnt/ssh.config /root/.ssh/config && service ssh start'"

-------

cd /mnt && mkdir librarys

git clone --recursive https://github.com/pytorch/pytorch

cd pytorch && git checkout v1.8.1 && git submodule sync && git submodule update --init --recursive

cd third_party/nccl && rm nccl -r && git clone https://github.com/parasailteam/nccl-master.git && mv nccl-master nccl && cd nccl 

change ProcessGroupNCCL

cd /mnt/librarys/pytorch && export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && python setup.py install

cd /mnt/librarys/ && git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

--------

all2all

https://github.com/pytorch/pytorch/blob/a4626348bc85b6484d647ceb07298b3516e2a7b3/torch/lib/c10d/ProcessGroupNCCL.cpp#L1563

all_reduce

https://github.com/pytorch/pytorch/blob/a4626348bc85b6484d647ceb07298b3516e2a7b3/torch/lib/c10d/ProcessGroupNCCL.cpp#L1249

extra processgroup 16 GPU

parallel-ssh -i -t 0 -p 200 -h pssh_2VM.host 'sudo pkill -9 python3'

parallel-ssh -i -t 0 -p 200 -h pssh_2VM.host 'sudo docker exec -i sccl_superbench bash -c "source env.sh && cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/root/nccl-master/build/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH  NCCL_ALGO=SCKL NCCL_PROTO=Simple SCKL_XML_FILE=/root/test.xml NCCL_NET_SHARED_BUFFERS=0   NCCL_MIN_NCHANNELS=16  NCCL_MAX_NCHANNELS=16  sh test_alltoall.sh"' 2>&1 | tee ./alltoall.log


parallel-ssh -i -t 0 -p 200 -h pssh_2VM.host 'sudo docker exec -i sccl_superbench bash -c "source env.sh && cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=6000  sh test_alltoall.sh"' 2>&1 | tee ./alltoall.log


parallel-ssh -i -t 0 -p 200 -h pssh_2VM.host 'sudo docker exec -i sccl_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=6000  sh ./examples/pretrain_gpt3_13B_2VMs.sh"' 2>&1 | tee ./13B_2VMs.log

pretrain_gpt3_13B_2VMs.sh

source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=6000  LD_PRELOAD=/opt2/msft/nccl-2.8.3-1/build/lib/libnccl.so:$LD_PRELOAD  sh ./examples/pretrain_gpt3_13B_2VMs.sh

parallel-ssh -i -t 0 -p 200 -h pssh_2VM.host 'sudo docker exec -i sccl_superbench bash -c "source env.sh && cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so:~/nccl-master/build/lib/libnccl.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH  NCCL_ALGO=SCKL  sh test_alltoall.sh"' 2>&1 | tee ./allreduce.log



 source /superbench/env.sh &  LD_PRELOAD=~/nccl-master/build/lib/libnccl.so  NCCL_DEBUG=INFO  NCCL_ALGO=SCKL NCCL_PROTO=Simple  NCCL_NET_SHARED_BUFFERS=0  NCCL_MIN_NCHANNELS=16  NCCL_MAX_NCHANNELS=16   SCKL_XML_FILE=~/test.xml   sh ./examples/pretrain_gpt_distributed_with_mp.sh 


 parallel-ssh -i -t 0 -h pssh.host "sudo docker run -it -d --name=sccl_dev --privileged --net=host --ipc=host --gpus=all -v $HOME/superbench-results:/superbench/Results -v /mnt:/mnt -v /mnt/tmp:/tmp -v /opt:/opt2 bench.azurecr.io/superbench:cuda11.1.1-20210204 bash"




 LD_LIBRARY_PATH=~/nccl-master/build/lib/ mpirun --allow-run-as-root -np 16 -H node0:8,node1:8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=SCKL --tag-output -x NCCL_NET_SHARED_BUFFERS=0 -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=16 -x SCKL_XML_FILE=~/test.xml   -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:~/nccl-master/build/lib/ -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml  -x NCCL_PROTO=Simple  ~/nccl-tests/build/alltoall_perf  -w 100 -n 100 -b 50M -e 300M -i 50000000 -c 1 -g 1 -z 0







 LD_LIBRARY_PATH=~/nccl-master/build/lib/ mpirun --allow-run-as-root -np 16 -H node0:8,node1:8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=SCKL --tag-output -x NCCL_NET_SHARED_BUFFERS=0 -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=16 -x SCKL_XML_FILE=~/test.xml   -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:~/nccl-master/build/lib/ -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml  -x NCCL_PROTO=Simple  ~/nccl-tests/build/all_reduce_perf  -w 100 -n 100 -b 50M -e 300M -i 50000000 -c 1 -g 1 -z 0


 [1,4]<stdout>:gpt3testlp3-000001:1140649:1140649 [4] enqueue.cc:323 NCCL WARN Error : no algorithm/protocol available
[1,4]<stdout>:gpt3testlp3-000001:1140649:1140649 [4] NCCL INFO enqueue.cc:405 -> 3
[1,4]<stdout>:gpt3testlp3-000001:1140649:1140649 [4] NCCL INFO enqueue.cc:513 -> 3
[1,4]<stdout>:gpt3testlp3-000001:1140649:1140649 [4] NCCL INFO enqueue.cc:713 -> 3
[1,4]<stdout>:gpt3testlp3-000001: Test NCCL failure all_reduce.cu:57 'internal error'
[1,4]<stdout>: .. gpt3testlp3-000001: Test failure common.cu:378
[1,4]<stdout>: .. gpt3testlp3-000001: Test failure common.cu:493
[1,4]<stdout>: .. gpt3testlp3-000001: Test failure all_reduce.cu:103
[1,4]<stdout>: .. gpt3testlp3-000001: Test failure common.cu:521
[1,4]<stdout>: .. gpt3testlp3-000001: Test failure common.cu:844



 parallel-ssh -i -t 0 -p 200 -h pssh_2VM.host 'sudo docker exec -i sccl_superbench bash -c "source env.sh && cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so:~/nccl-master/build/lib/libnccl.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH    sh test_allreduce.sh"' 2>&1 | tee ./allreduce.log


  parallel-ssh -i -t 0 -p 200 -h pssh_2VM.host 'sudo docker exec -i sccl_superbench bash -c "cd /superbench & source env.sh && cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=~/nccl-master/build/lib/libnccl.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH  NCCL_P2P_LEVEL=SYS  sh test_alltoall.sh"' 2>&1 | tee ./alltoall.log


  
mpirun --allow-run-as-root -np 16 -H node0:8,node1:8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0  --tag-output -x NCCL_NET_SHARED_BUFFERS=0 -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=16 -x SCKL_XML_FILE=~/test.xml   -x LD_PRELOAD=LD_LIBRARY_PATH=/root/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  -x LD_LIBRARY_PATH=/root/nccl-master/build/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:~/nccl-master/build/lib/ -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml  -x NCCL_PROTO=Simple  ~/nccl-tests/build/alltoall_perf  -w 100 -n 100 -b 50M -e 300M -i 50000000 -c 1 -g 1 -z 0



 source /superbench/env.sh && cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=~/nccl-master/build/lib/libnccl.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH  NCCL_P2P_LEVEL=SYS  sh test_alltoall.sh

 source /superbench/env.sh && cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=~/nccl-master/build/lib/libnccl.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH  NCCL_P2P_LEVEL=SYS  sh test_alltoall.sh


  source /superbench/env.sh && cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=~/nccl-master/build/lib/libnccl.so    NCCL_P2P_LEVEL=SYS   LD_LIBRARY_PATH=~/nccl-master/build/lib/  sh test_alltoall.sh

   source /superbench/env.sh && cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=~/nccl-master/build/lib/libnccl.so    NCCL_P2P_LEVEL=SYS    LD_LIBRARY_PATH=~/nccl-master/build/lib/ sh test_alltoall.sh


   cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=:~/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so    NCCL_P2P_LEVEL=SYS   LD_LIBRARY_PATH=~/nccl-master/build/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib   sh test_alltoall.sh

   cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=:~/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so    NCCL_P2P_LEVEL=SYS   LD_LIBRARY_PATH=~/nccl-master/build/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib   sh test_alltoall.sh

   PIPELINED_P2P_FUSION=1

   parallel-ssh -i -t 0 -p 200 -h pssh18VM.host 'sudo docker exec -i jun_superbench bash -c "source env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=18 NODE_RANK=$PSSH_NODENUM MASTER_ADDR=10.0.2.23 MASTER_PORT=12345   LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH LOG_NAME=p2pfusion_175B_18VM_mp_barrier MP_BARRIER=1 P2P_FUSION=1  sh ./examples/pretrain_gpt3_175B_18VMs.sh"' 2>&1 | tee ./GPT3_p2pfusion_175B_18VM_mp_barrier.log


  source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=~/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=~/nccl-master/build/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  LOG_NAME=p2pfusion_175B_2VM_mp_barrier MP_BARRIER=1  PIPELINED_P2P_FUSION=1  sh ./examples/pretrain_gpt3_13B_2VMs.sh

  source /superbench/env.sh &&  cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345   LD_PRELOAD=~/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=~/nccl-master/build/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib LOG_NAME=p2pfusion_175B_2VM_mp_barrier   MP_BARRIER=1 PIPELINED_P2P_FUSION=1  sh ./examples/pretrain_gpt3_13B_2VMs.sh

PIPELINED_P2P_FUSION=0 

NCCL_ALGO=

source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.33 MASTER_PORT=12345   LD_PRELOAD=/root/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/root/nccl-master/build/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  LOG_NAME=p2pfusion_175B_2VM_mp_barrier MP_BARRIER=1   sh ./examples/pretrain_gpt3_13B_2VMs.sh

source /superbench/env.sh &&  cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.33 MASTER_PORT=12345   LD_PRELOAD=/root/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/root/nccl-master/build/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib LOG_NAME=p2pfusion_175B_2VM_mp_barrier   MP_BARRIER=1   sh ./examples/pretrain_gpt3_13B_2VMs.sh


/opt2/msft/nccl-2.8.3-1/build/lib/libnccl.so

 mpirun --allow-run-as-root -np 16 -H node0:8,node1:8 -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0  --tag-output -x NCCL_NET_SHARED_BUFFERS=0 -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=16 -x SCKL_XML_FILE=~/test.xml   -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  -x LD_LIBRARY_PATH=~/nccl-master/build/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml  -x NCCL_PROTO=Simple  ~/nccl-tests/build/alltoall_perf  -w 100 -n 100 -b 50M -e 300M -i 50000000 -c 1 -g 1 -z 0 -d int8





 source /superbench/env.sh &&   cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 WORLD_SIZE=16 NODE_RANK=1 MASTER_ADDR=10.0.2.13  MASTER_PORT=12345    LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so      LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16  NCCL_NET_SHARED_BUFFERS=0 NCCL_DEBUG=INFO  SCKL_XML_FILE=~/test.xml NCCL_TOPO_FILE=/opt2/msft/topo.xml   NCCL_PROTO=Simple   sh test_pipelined_p2p_fusion.sh

 source /superbench/env.sh &&  cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 WORLD_SIZE=16  NODE_RANK=0 MASTER_ADDR=10.0.2.13  MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so       LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16  NCCL_NET_SHARED_BUFFERS=0 NCCL_DEBUG=INFO   SCKL_XML_FILE=~/test.xml NCCL_TOPO_FILE=/opt2/msft/topo.xml      NCCL_PROTO=Simple sh test_pipelined_p2p_fusion.sh



      cd /mnt/repos && GPUS_PER_NODE=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=10.0.2.33  MASTER_PORT=12345   LD_PRELOAD=:~/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so    NCCL_P2P_LEVEL=SYS   LD_LIBRARY_PATH=~/nccl-master/build/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1   sh test_pipelined_p2p_fusion.sh


sudo scp -i ~/.ssh/id_rsa  -r  pytorch     superbench@gpt3testlp3-000004:/home/superbench


 source /superbench/env.sh &&   cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 WORLD_SIZE=16 NODE_RANK=0 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345  NCCL_ALGO=RING  LD_PRELOAD=/opt2/msft/nccl-2.8.3-1/build/lib/libnccl.so    NCCL_P2P_LEVEL=SYS   LD_LIBRARY_PATH=/opt2/msft/nccl-2.8.3-1/build/lib  NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=16  NCCL_NET_SHARED_BUFFERS=0 NCCL_DEBUG=INFO  SCKL_XML_FILE=~/test.xml NCCL_TOPO_FILE=/opt2/msft/topo.xml   sh test_pipelined_p2p_fusion.sh

 source /superbench/env.sh &&  cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 WORLD_SIZE=16  NODE_RANK=1 MASTER_ADDR=10.0.2.11 MASTER_PORT=12345  NCCL_ALGO=RING LD_PRELOAD=/opt2/msft/nccl-2.8.3-1/build/lib/libnccl.so    NCCL_P2P_LEVEL=SYS   LD_LIBRARY_PATH=/opt2/msft/nccl-2.8.3-1/build/lib  NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=16  NCCL_NET_SHARED_BUFFERS=0 NCCL_DEBUG=INFO   SCKL_XML_FILE=~/test.xml NCCL_TOPO_FILE=/opt2/msft/topo.xml  sh test_pipelined_p2p_fusion.sh


 source /superbench/env.sh &&   cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 WORLD_SIZE=16 NODE_RANK=0 MASTER_ADDR=10.0.2.33 MASTER_PORT=12345    LD_PRELOAD=/root/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so      LD_LIBRARY_PATH=/root/nccl-master/build/lib:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16  NCCL_NET_SHARED_BUFFERS=0 NCCL_DEBUG=INFO  SCKL_XML_FILE=~/test.xml NCCL_TOPO_FILE=/opt2/msft/topo.xml  NCCL_ALGO=RING  NCCL_PROTO=Simple   sh test_pipelined_p2p_fusion.sh

 source /superbench/env.sh &&  cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 WORLD_SIZE=16  NODE_RANK=1 MASTER_ADDR=10.0.2.33 MASTER_PORT=12345   LD_PRELOAD=/root/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so       LD_LIBRARY_PATH=/root/nccl-master/build/lib:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16  NCCL_NET_SHARED_BUFFERS=0 NCCL_DEBUG=INFO   SCKL_XML_FILE=~/test.xml NCCL_TOPO_FILE=/opt2/msft/topo.xml   NCCL_ALGO=RING   NCCL_PROTO=Simple sh test_pipelined_p2p_fusion.sh


source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.33 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml LOG_NAME=p2pfusion_175B_2VM_mp_barrier MP_BARRIER=1  WARMUP_BATCH=64   sh ./examples/pretrain_gpt3_13B_2VMs.sh

source /superbench/env.sh &&  cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.33 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml  LOG_NAME=p2pfusion_175B_2VM_mp_barrier   MP_BARRIER=1  WARMUP_BATCH=64   sh ./examples/pretrain_gpt3_13B_2VMs.sh



source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.33 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml LOG_NAME=p2pfusion_175B_2VM_mp_barrier MP_BARRIER=1  WARMUP_BATCH=64  P2P_FUSION=1  sh ./examples/pretrain_gpt3_13B_2VMs.sh

source /superbench/env.sh &&  cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.33 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml  LOG_NAME=p2pfusion_175B_2VM_mp_barrier   MP_BARRIER=1  WARMUP_BATCH=64 P2P_FUSION=1  sh ./examples/pretrain_gpt3_13B_2VMs.sh



source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.33 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml LOG_NAME=p2pfusion_175B_2VM_mp_barrier MP_BARRIER=1  WARMUP_BATCH=64  P2P_SHARD=1  sh ./examples/pretrain_gpt3_13B_2VMs.sh

source /superbench/env.sh &&  cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.33 MASTER_PORT=12345   LD_PRELOAD=/usr/local/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/usr/local/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml  LOG_NAME=p2pfusion_175B_2VM_mp_barrier   MP_BARRIER=1  WARMUP_BATCH=64 P2P_SHARD=1  sh ./examples/pretrain_gpt3_13B_2VMs.sh




# pipelined p2p 
source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.33 MASTER_PORT=12345   LD_PRELOAD=/root/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/root/nccl-master/build/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml LOG_NAME=p2pfusion_175B_2VM_mp_barrier MP_BARRIER=1  PIPELINED_P2P_FUSION=1    WARMUP_BATCH=64   NCCL_PROTO=Simple  sh ./examples/pretrain_gpt3_13B_2VMs.sh

source /superbench/env.sh &&  cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.33 MASTER_PORT=12345   LD_PRELOAD=/root/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/root/nccl-master/build/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml  LOG_NAME=p2pfusion_175B_2VM_mp_barrier   MP_BARRIER=1  PIPELINED_P2P_FUSION=1  WARMUP_BATCH=64 NCCL_PROTO=Simple  sh ./examples/pretrain_gpt3_13B_2VMs.sh

# p2p meg

source /superbench/env.sh && cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.2.11  MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml LOG_NAME=p2pfusion_175B_2VM_mp_barrier MP_BARRIER=1  P2P_MEG=1    WARMUP_BATCH=64   NCCL_PROTO=Simple  sh ./examples/pretrain_gpt3_13B_2VMs.sh

source /superbench/env.sh &&  cd /mnt/repos/Megatron-LMx && GPUS_PER_NODE=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.2.11  MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so  LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib/:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib   NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16   NCCL_NET_SHARED_BUFFERS=0  SCKL_XML_FILE=~/test.xml  LOG_NAME=p2pfusion_175B_2VM_mp_barrier   MP_BARRIER=1  P2P_MEG=1   WARMUP_BATCH=64 NCCL_PROTO=Simple  sh ./examples/pretrain_gpt3_13B_2VMs.sh



        typechange: .dockerignore
        modified:   .gitmodules
        typechange: caffe2/python/examples/resnet50_trainer.py
        modified:   third_party/nccl/nccl (new commits)
        modified:   torch/csrc/distributed/c10d/init.cpp
        modified:   torch/distributed/distributed_c10d.py
        modified:   torch/lib/c10d/ProcessGroup.cpp
        modified:   torch/lib/c10d/ProcessGroup.hpp
        modified:   torch/lib/c10d/ProcessGroupNCCL.cpp
        modified:   torch/lib/c10d/ProcessGroupNCCL.hpp

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        test/docker_setup_and_run_alltoall.sh
        test/test_allreduce.py
        test/test_allreduce.sh
        test/test_alltoall.py
        test/test_alltoall.sh
        test/test_pipelined_p2p_fusion.py
        test/test_pipelined_p2p_fusion.sh




source /superbench/env.sh &&   cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 WORLD_SIZE=16 NODE_RANK=0 MASTER_ADDR=10.0.2.11  MASTER_PORT=12345    LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so      LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16  NCCL_NET_SHARED_BUFFERS=0 NCCL_DEBUG=INFO  SCKL_XML_FILE=~/test.xml NCCL_TOPO_FILE=/opt2/msft/topo.xml   NCCL_PROTO=Simple   sh test_pipelined_p2p_fusion.sh

source /superbench/env.sh &&  cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 WORLD_SIZE=16  NODE_RANK=1 MASTER_ADDR=10.0.2.11  MASTER_PORT=12345   LD_PRELOAD=/mnt/librarys/pytorch/build/nccl/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so       LD_LIBRARY_PATH=/mnt/librarys/pytorch/build/nccl/lib:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16  NCCL_NET_SHARED_BUFFERS=0 NCCL_DEBUG=INFO   SCKL_XML_FILE=~/test.xml NCCL_TOPO_FILE=/opt2/msft/topo.xml      NCCL_PROTO=Simple sh test_pipelined_p2p_fusion.sh



 source /superbench/env.sh &&   cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 WORLD_SIZE=16 NODE_RANK=0 MASTER_ADDR=10.0.2.33  MASTER_PORT=12345    LD_PRELOAD=/root/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so      LD_LIBRARY_PATH=/root/nccl-master/build/lib:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16  NCCL_NET_SHARED_BUFFERS=0 NCCL_DEBUG=INFO  SCKL_XML_FILE=~/test.xml NCCL_TOPO_FILE=/opt2/msft/topo.xml   NCCL_PROTO=Simple   sh test_pipelined_p2p_fusion.sh

 source /superbench/env.sh &&  cd /mnt/repos && GPUS_PER_NODE=8 NNODES=2 WORLD_SIZE=16  NODE_RANK=1 MASTER_ADDR=10.0.2.33  MASTER_PORT=12345   LD_PRELOAD=/root/nccl-master/build/lib/libnccl.so:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so       LD_LIBRARY_PATH=/root/nccl-master/build/lib:/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16  NCCL_NET_SHARED_BUFFERS=0 NCCL_DEBUG=INFO   SCKL_XML_FILE=~/test.xml NCCL_TOPO_FILE=/opt2/msft/topo.xml      NCCL_PROTO=Simple sh test_pipelined_p2p_fusion.sh  > log.txt



mpirun --allow-run-as-root -np 128 -H node0:8,node1:8,node2:8,node3:8,node4:8,node5:8,node6:8,node7:8,node8:8,node9:8,node10:8,node11:8,node12:8,node13:8,node14:8,node15:8  -mca pml ob1 --mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x NCCL_ALGO=Ring -x LD_PRELOAD=/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib/libnccl-net.so -x PATH -x LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt2/msft/nccl-rdma-sharp-plugins-2.8.3/lib:$LD_LIBRARY_PATH -x UCX_IB_ENABLE_CUDA_AFFINITY=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x UCX_IB_PCI_RELAXED_ORDERING=on -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt2/msft/topo.xml /nccl-tests/build/all_reduce_perf -b 1K -e 8G -f 2 -g 1 -c 0 -n 50