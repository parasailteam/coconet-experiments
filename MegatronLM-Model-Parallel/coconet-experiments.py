import os, sys
import re
from subprocess import getstatusoutput
from string import Template

master_addr = "127.0.0.1"
master_port = "10000"
gpt2_training_data = sys.argv[1]
bert_training_data = sys.argv[2]

nproc = os.environ.get("NPROC")

impls = ["AR-C", "overlap"]

def execute_command(command):
    (s, o) = getstatusoutput(command)
    print(o)
    if s != 0:
        assert False, "command '%s' did not execute successfully\n '%s'"%(command, o)
    return o

def get_result(output):
    times = re.findall(r'forward:\s*([\d\.]+)', output)
    times = sorted([float(t) for t in times])
    if len(times) > 1:
        times = times[1:max(1000, len(times)-1)]

    return sum(times)/len(times)

def execute_command_and_get_result(impl, model, training_data):
    megatron_env = 'NPROC="%s" MEGATRON=\"%s\" MEGATRON_LAYERNORM=\"REPEATED\"'%(nproc, impl)
    if model == "gpt2":
        command = megatron_env + " ./examples/pretrain_gpt2_1200M_distributed.sh %s"%(training_data)
    elif model == "bert":
        command = megatron_env + " ./examples/pretrain_bert_distributed.sh %s"%(training_data)
    o = execute_command(command)
    return get_result(o)

gpt2_baseline = execute_command_and_get_result("AR-C", "gpt2", gpt2_training_data)
gpt2_overlap = execute_command_and_get_result("overlap", "gpt2", gpt2_training_data)

bert_baseline = execute_command_and_get_result("AR-C", "bert", bert_training_data)
bert_overlap = execute_command_and_get_result("overlap", "bert", bert_training_data)

print ("\n\nModel Parallel Inference Improvements:")
print ("GPT2 Imprvement: ", gpt2_baseline/gpt2_overlap)
print ("BERT Imprvement: ", bert_baseline/bert_overlap)