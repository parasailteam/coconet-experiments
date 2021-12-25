/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include <assert.h>

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, recvbuff, NULL, NULL, NULL, nullptr, nullptr, nullptr, nullptr, nullptr,0, 
    /*ReduceScattered Params*/ nullptr, nullptr, nullptr,nullptr,nullptr,  0, nullptr, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceScatteredWeights, const void* sendbuff, void** weightbuff,
    size_t nbuff, size_t* counts, size_t totalCount,  void* alpha, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceScatteredWeights(const void* sendbuff, void** weightbuff, size_t nbuff, size_t* counts, size_t totalCount,  void* alpha,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, nullptr, nullptr, nullptr, alpha, nullptr,nullptr, nullptr, nullptr, nullptr, 0, 
    /*ReduceScattered Params*/ nullptr, weightbuff, nullptr, nullptr, nullptr, nbuff, counts,
    /*Other args*/ totalCount, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceScatteredGradWeights, const void** sendbuff, void** weightbuff,
    size_t nbuff, size_t* counts, size_t totalCount,  void* alpha, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceScatteredGradWeights(const void** sendbuff, void** weightbuff, size_t nbuff, size_t* counts, size_t totalCount,  void* alpha,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  struct ncclInfo info = { ncclCollAllReduceScattered, "AllReduceScattered",
    nullptr, nullptr, nullptr, nullptr, alpha, nullptr, nullptr,nullptr, nullptr, nullptr, 0, 
    /*ReduceScattered Params*/ sendbuff, weightbuff, nullptr, nullptr, nullptr, nbuff, counts,
    /*Other args*/ totalCount, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceScatteredAdamMP, const void** sendbuff, void** weightbuff, float** floatWeightBuff, void* firstMomentBuff, void* secondMomentBuff,
    size_t nbuff, size_t* counts, size_t totalCount, void* alpha, void* beta1Buff, void* beta2Buff, void* unscaleParameterBuff, int* numOverflows, const int epoch, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceScatteredAdamMP(const void** sendbuff, void** weightbuff, float** floatWeightBuff, void* firstMomentBuff, void* secondMomentBuff, size_t nbuff, size_t* counts, size_t totalCount,
    void* alpha, void* beta1Buff, void* beta2Buff, void* unscaleParameterBuff, int* numOverflows, const int epoch,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollAllReduceScattered, "AllReduceScattered",
    nullptr, nullptr, nullptr, nullptr, alpha, nullptr, nullptr, 
    beta1Buff, beta2Buff, unscaleParameterBuff, epoch, 
    /*ReduceScattered Params*/ sendbuff, weightbuff, floatWeightBuff, firstMomentBuff, secondMomentBuff, nbuff, counts,
    totalCount, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, OptimizerType::Adam, nullptr, 0, nullptr, nullptr , nullptr, numOverflows};
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceScatteredLAMB, const void** sendbuff, void** weightbuff, float** floatWeightBuff, void* firstMomentBuff, void* secondMomentBuff,
    size_t smallNbuff, size_t* counts, size_t totalCount,  const size_t* parentBuffSizes, void* alpha, void* beta1Buff, void* beta2Buff, void* unscaleParameterBuff, int* numOverflows, void* weightNormBuff, void* rStorageBuff, 
    const size_t nbuff, size_t* buffIdToParentBufferId, const int epoch, ncclDataType_t datatype, 
    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceScatteredLAMB(const void** sendbuff, void** weightbuff, float** floatWeightBuff, void* firstMomentBuff, void* secondMomentBuff, size_t smallNbuff, size_t* counts, size_t totalCount,  const size_t* parentBuffSizes,
    void* alpha, void* beta1Buff, void* beta2Buff, void* unscaleParameterBuff, int* numOverflows, void* weightNormBuff, void* rStorageBuff, const size_t nBuff, size_t* buffIdToParentBufferId, const int epoch,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollAllReduceScattered, "AllReduceScattered",
    nullptr, nullptr, nullptr, nullptr, alpha, nullptr, nullptr, 
    beta1Buff, beta2Buff, unscaleParameterBuff, epoch, 
    /*ReduceScattered Params*/ sendbuff, weightbuff, floatWeightBuff,firstMomentBuff, secondMomentBuff, smallNbuff, counts,
    totalCount, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, OptimizerType::LAMB, weightNormBuff, nBuff, buffIdToParentBufferId, rStorageBuff, parentBuffSizes, numOverflows};
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduce2, const void* sendbuff, void* recvbuff, void* weightbuff,
    size_t count,  void* alpha, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce2(const void* sendbuff, void* recvbuff, void* weightbuff, size_t count, void* alpha,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  
  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, nullptr, weightbuff, nullptr, alpha, nullptr, nullptr, nullptr, nullptr,nullptr, 0, 
    /*ReduceScattered Params*/ nullptr, nullptr, nullptr, nullptr, nullptr,0, nullptr,
    count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceAdam, const void* sendbuff, void* recvbuff, void* weightbuff,
    size_t count,  void* alpha, void* firstMomentBuff, void* secondMomentBuff, void* beta1Buff, void* beta2Buff,
    const int epoch, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceAdam(const void* sendbuff, void* recvbuff, void* weightbuff, size_t count, void* alpha,
    void* firstMomentBuff, void* secondMomentBuff, void* beta1Buff, void* beta2Buff, const int epoch, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  
  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, nullptr, weightbuff, nullptr, alpha, firstMomentBuff, secondMomentBuff, 
    beta1Buff, beta2Buff, nullptr,epoch, 
    /*ReduceScattered Params*/ nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr,
    count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
