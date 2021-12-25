import os, sys
import re
from subprocess import getstatusoutput
from string import Template

bert_configs = ["334M", "1.2B", "3.9B"]

master_addr = "127.0.0.1"
master_port = "10000"
training_data = sys.argv[1]

nproc = os.environ.get("NPROC")

impls = ["NV_BERT", "PyTorch_DDP", "CoCoNet"]

impl_to_batch_size = {"adam":{"334M": {"NV_BERT": 32, "PyTorch_DDP": 32, "CoCoNet":32}, 
                              "1.2B": {"NV_BERT": 8, "PyTorch_DDP": 8, "CoCoNet":32},
                              "3.9B":  {"NV_BERT": 2, "PyTorch_DDP": 2, "CoCoNet":8}},

                      "lamb":{"334M": {"NV_BERT": 64, "PyTorch_DDP": 64, "CoCoNet":128}, 
                              "1.2B": {"NV_BERT": 8, "PyTorch_DDP": 8, "CoCoNet":64},
                              "3.9B":  {"NV_BERT": 2, "PyTorch_DDP": 2, "CoCoNet":8}}
                     }

global_batch_sizes = {"adam" : 8192, "lamb": 65536}

def gradient_accumulation_steps(global_batch_size, per_gpu_batch_size_256_gpus):
    total_per_gpu_batch_size = global_batch_size/int(nproc)
    per_step_batch_size = min(total_per_gpu_batch_size, per_gpu_batch_size_256_gpus)
    grad_accum_steps = total_per_gpu_batch_size/per_step_batch_size
    return int(grad_accum_steps),int(total_per_gpu_batch_size)

def impl_to_command_arg(impl):
    if (impl == "NV_BERT"):
        return "--fp16 --allreduce_post_accumulation --allreduce_post_accumulation_fp16"
    if (impl == "PyTorch_DDP"):
        return "--fp16 --allreduce_post_accumulation --allreduce_post_accumulation_fp16"
    if (impl == "CoCoNet"):
        return "--fused_allreduce --fused_allreduce_fp16"

def get_bert_command(bert_config, impl, optimizer):
    mpirun = Template("mpirun -np $nproc -x NCCL_MIN_NCHANNELS=32 -x NCCL_NTHREADS=640 -x NCCL_ALGO=Ring -x NCCL_PROTO=Simple -x NCCL_MAX_NCHANNELS=32  -x NCCL_BUFFSIZE=4194304 -x MASTER_ADDR=$master_addr -x MASTER_PORT=$master_port")
    bert_command = Template("""python3 run_pretraining.py --input_dir=$training_data --output_dir=./checkpoints/ --config_file=$bert_config --bert_model=bert-large-uncased --train_batch_size=$batch_size \
                    --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=$max_steps --num_train_epochs=1 --warmup_proportion=0.2843 --learning_rate=6e-3 --seed=12439 --gradient_accumulation_steps=$grad_accum_steps --do_train \
                    $optimizer --num_steps_per_checkpoint=1000000 $impl_args""")
    
    batch_size = impl_to_batch_size[optimizer.lower()][bert_config][impl]
    grad_accum_steps, batch_size = gradient_accumulation_steps(global_batch_sizes[optimizer.lower()], batch_size)
    max_steps = 5
    impl_args = impl_to_command_arg(impl)
    mpirun = mpirun.substitute({"nproc": nproc, "master_addr":master_addr, "master_port": master_port})
    optimizer = "" if optimizer.lower() == "lamb" else "--use_adam"
    bert_command = bert_command.substitute({"bert_config":"bert_config_%s.json"%(bert_config), "training_data": training_data, "batch_size": int(batch_size), "optimizer": optimizer, "impl_args": impl_args, "max_steps": int(max_steps), "grad_accum_steps": int(grad_accum_steps)})

    print (mpirun + " " + bert_command)

    return mpirun + " " + bert_command, grad_accum_steps, batch_size

def execute_command(command):
    (s, o) = getstatusoutput(command)

    if s != 0:
        if o.lower().find("CUDA out of memory"):
            return "OOM"
        assert False, "command '%s' did not execute successfully\n '%s'"%(command, o)
    return o

def get_result(output):
    if (output == "OOM"):
        return "OOM"
    times = re.findall(r'global step duration\s*([\d\.]+)', output)
    times = sorted([float(t) for t in times])
    if len(times) > 1:
        times = times[1:max(1000, len(times)-1)]

    return sum(times)/len(times)

def execute_command_and_get_result(bert_config, impl, optimizer):
    c, grad_accum_steps, batch_size = get_bert_command(bert_config, impl, optimizer)
    o = execute_command(c)
    print (o)
    return get_result(o)

adam_results = {c : {i : -1 for i in impls} for c in bert_configs}
lamb_results = {c : {i : -1 for i in impls} for c in bert_configs}

for config in adam_results:
    for impl in adam_results[config]:
        adam_results[config][impl] = execute_command_and_get_result(config, impl, "Adam")

for config in lamb_results:
    for impl in lamb_results[config]:
        lamb_results[config][impl] = execute_command_and_get_result(config, impl, "LAMB")

print (adam_results)
print (lamb_results)

print ("Table 1: Maximum Micro Batch Size supported by all implementations and speedup of COCONET over the baselines when training BERT model with three different parameter configurations using Adam and LAMB optimizer. OOM means Out of Memory")

def print_results(optim, results):
    print ("-------- Results for %s ---------"%optim)
    row_format = "{:>15}" * 6
    print (row_format.format("Parameters", "", "Maximum Micro Batch Size", "", "", "Speedup of CoCoNet"))
    print (row_format.format(*([""] + impls + sorted(["NV BERT", "PyTorch_DDP"]))))

    for config in results:
        config_row = [config]
        for impl in impls:
            if results[config][impl] == "OOM":
                config_row += ["-"]
            else:
                config_row += [impl_to_batch_size["adam"][config][impl]]
        for impl in impls:
            if results[config]["CoCoNet"] == "OOM":
                config_row += ["-"]
            elif impl != "CoCoNet":
                if results[config][impl] == "OOM":
                    config_row += ["-"]
                else:
                    f = "%.2f"%(results[config][impl]/results[config]["CoCoNet"])
                    config_row += [f]
        print (row_format.format(*config_row))

print_results("Adam", adam_results)
print_results("LAMB", lamb_results)