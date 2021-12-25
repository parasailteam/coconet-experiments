/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "core.h"

typedef enum {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollTreeUp,
  ncclPatternCollTreeDown
} ncclPattern_t;

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  void* weightbuff;
  void* newweightbuff;
  void* alpha;
  //Adam Parameters
  void* firstMomentBuff; 
  void* secondMomentBuff;
  void* beta1Buff;
  void* beta2Buff;
  void *unscaleParameterBuff;
  int epoch;
  //Scattered Pointer Params
  const void** scatteredSendbuff;
  void** scatteredWeightbuff;
  float** scatteredFloatWeightBuff;
  void* scatteredFirstMomentBuff;
  void* scatteredSecondMomentBuff;
  size_t scatteredSmallNBuff;
  size_t* scatteredBuffSizes;
  ///
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root;
  ncclComm_t comm;
  cudaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;

  OptimizerType optimizerType;
  void* weightNormBuff;
  size_t nbuff;
  size_t* buffIdToParentBufferId;
  void* rStorageBuff;
  const size_t* parentBuffSizes;
  int* numOverflows;

  // Computed later
  int algorithm;
  int protocol;
  ncclPattern_t pattern;
  int nChannels;
  int nThreads;
  size_t nBytes;
  int nstepsPerLoop;
  int nchunksPerLoop;

};

#endif
