# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math
import os

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .mappings import copy_to_model_parallel_region
from .mappings import gather_from_model_parallel_region
from .mappings import reduce_from_model_parallel_region
from .mappings import scatter_to_model_parallel_region
from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility

from megatron import mpu

execution_type = os.getenv('MEGATRON')
assert(execution_type in ['AR-C', 'RS-C-AG', 'fuse(RS-C-AG)', 'overlap'])
layernorm_execution_type = os.getenv('MEGATRON_LAYERNORM')
assert(layernorm_execution_type in ['REPEATED', 'C-AG', 'fuse(C-AG)'])

def _initialize_affine_weight(weight, output_size, input_size,
                              per_partition_size, partition_dim, init_method,
                              stride=1, return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.stride = stride

    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=weight.dtype,
                                requires_grad=False)
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_model_parallel_rank(),
                get_model_parallel_world_size())
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings_per_partition,
                                             self.embedding_dim))
        # And initialize.
        _initialize_affine_weight(
            self.weight, self.num_embeddings, self.embedding_dim,
            self.num_embeddings_per_partition, 0, init_method)

    def forward(self, input_):
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_model_parallel_region(output_parallel)
        return output


class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_,
                 keep_master_weight_for_test=False):
        super(ParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set some detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        # Divide the weight matrix along the embedding dimension.
        world_size = get_model_parallel_world_size()
        self.embedding_dim_per_partition = divide(self.embedding_dim,
                                                  world_size)

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings,
                                             self.embedding_dim_per_partition))
        # And initialize.
        _initialize_affine_weight(
            self.weight, self.num_embeddings, self.embedding_dim,
            self.embedding_dim_per_partition, 1, init_method,
            stride=1, return_master_weight=False)

    def forward(self, input_):
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = F.embedding(input_parallel, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        output = gather_from_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition,
                                             self.input_size))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight, self.output_size, self.input_size,
            self.output_size_per_partition, 0, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output

import os
from torch.autograd import Function
import math
#import pos_cuda
from megatron import get_timers
import torch.distributed as dist

'''
idIsInit = False
theId = torch.zeros([128], dtype=torch.int, device="cuda:"+str(torch.distributed.get_rank()))

def getCommId():
    if (idIsInit == False):
        if(torch.distributed.get_rank() == 0):
                theId = pos_cuda.get_id(theId)
        torch.distributed.broadcast(theId,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
        idIsInit = True
    return theId
''' 

tensorForSize = {}

class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 apply_residual_connection_post_layernorm = False,
                 layer_number = 0):
        super(RowParallelLinear, self).__init__()        

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.layer_number = layer_number
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        
        self.weight = Parameter(torch.Tensor(self.output_size,
                                             self.input_size_per_partition))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight, self.output_size, self.input_size,
            self.input_size_per_partition, 1, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)

        if execution_type == 'RS-C-AG':
            self.dropout = torch.nn.Dropout(0.1)

        batch_size = 32
        output_size = [batch_size, 1024, self.weight.size()[0]]
        # print("output_size", output_size, self.weight.dtype)
        # self.output_parallel = torch.zeros(output_size, dtype = torch.half, device="cuda", requires_grad=True)
        # self.register_buffer("output_parallelBuff", self.output_parallel)

    def backward(self, grad_output):
        grad_bias = grad_output.sum(dim=0).sum(dim=0)
        #timers('bias in backward').stop()
        
        #torch.cuda.synchronize()

        #grad_output = grads[0]
        #grad_bias = grads[1]
        #print(grad_output.shape)
        #print(grad_bias.shape)
        return grad_output, grad_bias, None

    def forward(self, input_, add_ten):
        #os.environ["NCCL_COMM_ID"] = "localhost:12101"
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        # print(input_parallel.shape,self.weight.shape)
        # timers("MatMul").start()
        if (execution_type == "overlap"):

            if False:                
                output_parallel = self.output_parallel.detach()
            else:
                output_size = [input_parallel.size()[0], input_parallel.size()[1], self.weight.size()[0]]
                output_parallel = torch.empty(output_size, dtype = input_parallel.dtype, device="cuda", requires_grad = True)

            dist.overlap_matmul_allreduce_bias_dropout_layernorm([input_parallel], [self.weight], [self.bias], 
                                                                  [add_ten], [output_parallel], output_parallel.shape[2])
            return output_parallel
        else:
            output_parallel = F.linear(input_parallel, self.weight)
            # timers("MatMul").stop()
            #grads = pos_cuda.backward(grad_output, D[0])
            #grad_output_d = grads[0]
            # print(output_parallel.shape)
            # All-reduce across all the partitions.
            if (execution_type == 'AR-C'): 
                # 
                # timers('AllReduce+Comp').start()
                output_ = reduce_from_model_parallel_region(output_parallel)
                if self.bias is not None:
                    output = output_ + self.bias
                else:
                    output = output_
                
                # timers('AllReduce+Comp').stop()
            elif execution_type == 'RS-C-AG':
                # print(output_parallel.shape, self.bias.shape, add_ten.shape)       
                # raise NotImplementedError("RS-C-AG is not implemented currently")
                if False:
                    output_parallel = reduce_from_model_parallel_region(output_parallel)
                    output_parallel = self.dropout(output_parallel)
                    if self.bias is not None:
                        output_parallel = output_parallel + self.bias
                    else:
                        output = output_parallel
                    
                    output = output_parallel + add_ten
                else:
                    # print(382, hex(add_ten.data_ptr()))
                    if self.bias is not None:
                        if self.apply_residual_connection_post_layernorm and layernorm_execution_type != 'REPEATED':
                            dist.model_parallel_update_with_layer_norm([output_parallel], [self.bias], [add_ten], output_parallel.shape[2])
                        else:
                            dist.model_parallel_update([output_parallel], [self.bias], [add_ten])
                            
                        output = output_parallel
                    else:
                        output = output_
                        
            else: # execution_type is 'fuse(RS-C-AG)':
                raise NotImplementedError("fuse(RS-C-AG) is not implemented currently")
                #output_ = reduce_from_model_parallel_region(output_parallel)
                if self.bias is not None:
                    #print(output_.shape)
                    #print(self.bias.shape)
                    #print(self.bias[0])
                    #output = output_ + self.bias
                    #output = FusedPOSFunction.apply(output_, self.bias, add_ten)
                    #theId = torch.zeros([128], dtype=torch.int, device="cuda:"+str(torch.distributed.get_rank()))
                    #if(torch.distributed.get_rank() == 0):
                    #    theId = pos_cuda.get_id(theId)
                    #torch.distributed.broadcast(theId,
                    #                            mpu.get_model_parallel_src_rank(),
                    #                            group=mpu.get_model_parallel_group())
                    #print(theId)
                    output = FusedPOSFunction.apply(output_parallel, self.bias, add_ten)
                else:
                    output = output_

        return output


class FusedPOSFunction(Function):
    
    '''
    def __init__(self, input):
        super(FusedPOSFunction, self).__init__()
        #self.stream = torch.cuda.Stream()
        #self.D = gen_d(input)

    
    def gen_d(input):
        shape = torch.Size((input.size()[0], input.size()[1], input.size()[2]))
        D = torch.zeros_like(input)
        torch.rand(shape, out=D)
        O = D > 0.5
        D = O * torch.ones_like(D) 
        return D           
    '''

    @staticmethod
    def forward(ctx, input, bias, add_ten):

        '''
        global D, firstTime, stream
        '''

        #timers = get_timers()
        '''
        timers('random generation').start()
        shape = torch.Size((input.size()[0], input.size()[1], input.size()[2]//32))
        D = torch.cuda.IntTensor(shape)
        torch.randint(-2**31, 2**31, shape, out=D)
        timers('random generation').stop()
        '''      

        #timers('random generation').start()
        #shape = torch.Size((input.size()[0], input.size()[1], input.size()[2]))
        #print(shape)
        #D = torch.zeros_like(input).uniform_(0, 1)
        
        '''
        if (firstTime):
            D = torch.randint(2, size = torch.Size((input.size()[0], input.size()[1], input.size()[2])), dtype=torch.float16,device=input.device)
            #bitD = torch.randint(-2**31, 2**31, size = torch.Size((64, 512, 1024//32)), dtype=torch.int, device=input.device)
            #D = torch.empty(64, 512, 1024, dtype=torch.float16, device=input.device)
            #pos_cuda.bit_to_fp16(D, bitD)
            firstTime = False
            stream = torch.cuda.Stream(device=input.device)#+str(torch.distributed.get_rank())))
        '''

        #torch.rand(shape, out=D)
        #O = D > 0.5
        #D = O * torch.ones_like(D)
        #D = torch.bernoulli(D)
        #timers('random generation').stop() 
        #print(D[0][0][0:10])

        #torch.cuda.synchronize(self.stream)     
        #D = self.D
        '''
        ctx.save_for_backward(D)
        '''
        #output = pos_cuda.forward(input, bias, D, add_ten)

        #print(add_ten[0][0][0].data)

        #if (torch.distributed.get_rank() == 0):
        #    print("\n\n", input[0][0][0])
        #input_ref = reduce_from_model_parallel_region(input)
        #torch.cuda.synchronize()
        #if (torch.distributed.get_rank() == 0):
        #    print(input[0][0][0], "\n\n")
        #if (torch.distributed.get_rank() == 0):
        #    print("ref: " , (input_ref[0][0][0]+bias[0])*D[0][0][0] + add_ten[0][0][0])
        #    print("resid: ", add_ten[0][0][0])
        
        #timers('DSL pos Forward').start()
        #torch.cuda.synchronize()
        #output = pos_cuda.forward(input, bias, D, add_ten)

        '''
        x1 = torch.zeros([10,10,10], dtype=torch.float16, device="cuda:"+str(torch.distributed.get_rank()))
        x2 = torch.zeros([10], dtype=torch.float16, device="cuda:"+str(torch.distributed.get_rank()))
        x3 = torch.zeros([10,10,10], dtype=torch.float16, device="cuda:"+str(torch.distributed.get_rank()))
        x4 = torch.zeros([10,10,10], dtype=torch.float16, device="cuda:"+str(torch.distributed.get_rank()))
        for i in range(0, 10):
            x2[i] = 2.0
            for j in range(0, 10):
                for k in range(0, 10):
                    x1[i][j][k] = 1.0
                    x3[i][j][k] = 3.0
                    x4[i][j][k] = 4.0
        temp = pos_cuda.forward_flatten(x1.view(-1), x2.view(-1).repeat(10*10), x3.view(-1), x4.view(-1), torch.distributed.get_world_size(), torch.distributed.get_rank())
        if (torch.distributed.get_rank() == 0):
            print(temp[0][0])
        '''
        
        '''
        stream.synchronize()  
        '''

        #output = pos_cuda.forward_flatten(input.view(-1), bias.view(-1).repeat(input.size()[0]*input.size()[1]), D.view(-1), add_ten.view(-1), torch.distributed.get_world_size(), torch.distributed.get_rank())
        #temp = pos_cuda.forward_flatten(input.view(-1), bias.view(-1).repeat(input.size()[0]*input.size()[1]), D.view(-1), add_ten.view(-1), torch.distributed.get_world_size(), torch.distributed.get_rank())
        dist.model_parallel_update([input], [bias], [add_ten])

        #print(type(torch.distributed.sgd_udpate))
        #print(type(torch.distributed.is_available))
        #torch.cuda.synchronize()

        #if (torch.distributed.get_rank() == 0):
        #    print("resid after: ", add_ten[0][0][0])
        #    print(output[0][0], "\n\n\n\n")

        #torch.cuda.synchronize()

        #timers('DSL pos Forward').stop()
        #del os.environ['NCCL_COMM_ID']

        #print(output[0].data)

        #print(input[0][0][0])
        #input_ = reduce_from_model_parallel_region(input)
        #output = (input_ + bias) * D + add_ten

        '''
        with torch.cuda.stream(stream):
            D = torch.randint(2, size = torch.Size((input.size()[0], input.size()[1], input.size()[2])), dtype=torch.float16, device = input.device )
            #bitD = torch.randint(-2**31, 2**31, size = torch.Size((64, 512, 1024//32)), dtype=torch.int,device=input.device)
            #D = torch.empty(64, 512, 1024, dtype=torch.float16, device=input.device)
            #pos_cuda.bit_to_fp16(D, bitD)
        '''

        #torch.cuda.synchronize()

        #print(type(output[0]))
        #return output[0].view(input.size()[0], input.size()[1], input.size()[2])
        #return output
        return input

    @staticmethod
    def backward(ctx, grad_output):
        #timers = get_timers()
        '''
        D = ctx.saved_tensors
        #torch.cuda.synchronize()
        #print(grad_output.size(), D[0].size())
        #print(torch.distributed.get_rank(), torch.distributed.get_world_size())
        timers('dropout in backward').start()
        #grads = pos_cuda.backward(grad_output, D[0])
        #grad_output_d = grads[0]
        grad_output_d = D[0] * grad_output
        timers('dropout in backward').stop()
        '''

        #torch.cuda.synchronize()
        
        #timers('bias in backward').start()
        '''
        grad_bias = grad_output_d.sum(dim=0).sum(dim=0)
        '''
        grad_bias = grad_output.sum(dim=0).sum(dim=0)
        #timers('bias in backward').stop()
        
        #torch.cuda.synchronize()

        #grad_output = grads[0]
        #grad_bias = grads[1]
        #print(grad_output.shape)
        #print(grad_bias.shape)
        return grad_output, grad_bias, None


'''
class FusedPOS(torch.nn.Module):
    def __init__(self, add_ten):
        super(FusedPOS, self).__init__()
        self.add_ten = add_ten

    def forward(self, input):
        X = input
        X = X.matmul(self.weights)
        #X = X + self.bias
        #X = AddBias.apply(X, self.bias)
        #print(X.shape)
        #print(self.bias.shape)
        #X = AddBiasCPU.apply(X, self.bias)
        X = AddBiasGPU.apply(X, self.bias)
        return X
'''
