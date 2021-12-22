/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include <assert.h>
#include <curand_kernel.h>

#define LALA(__step) do { \
        if (isMatMulOverlap && threadIdx.x == 0 && offset <= 0 && offset + nelem >= 0) {\
          printf("step %d nelem %d offset %ld rank %d bid %d %f\n", __step, nelem, offset, comm->rank, bid, (float)matrixM1M2[0]);\
        }\
} while(0);

#define SyncChildKernel do{}while(0);


template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceRingKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads-WARP_SIZE;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;


  // Compute pointers
  // const T ** __restrict__ thisScatteredInput = (const T*)args->ThisScatteredSendBuff;
  // const T ** __restrict__ thisScatteredWeight = (const T*)args->ThisScatteredSendBuff;
  // if (thisScatteredInput != nullptr and thisScatteredWeight != nullptr) {
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  T * __restrict__ thisWeight = (T*)args->ThisWeight;
  volatile int* thisSyncGlobalMem = (volatile int*)args->SyncGlobalMem;
  const T * __restrict__ thisAlpha = (T*)args->ThisAlpha;
  const T alpha = (thisAlpha == NULL) ? (T)0.0 : *thisAlpha;
  const T beta1 = (args->ThisBeta1 == NULL) ? (T)0.0 : *(T*)args->ThisBeta1;
  const T beta2 = (args->ThisBeta1 == NULL) ? (T)0.0 : *(T*)args->ThisBeta2;
  T* __restrict__ thisFirstMoment = (T*)args->ThisFirstMoment;
  T* __restrict__ thisSecondMoment = (T*)args->ThisSecondMoment;
  const int epoch = args->epoch;
  const bool doAdam = (thisFirstMoment != NULL && thisSecondMoment != NULL);
  OptimizerType optimizerType = args->optimizerType;

  const T * __restrict__ bias = (T*)args->ThisFirstMoment;
  const T * __restrict__ addTensor = (T*)args->ThisSecondMoment;
  int biasSize = args->epoch;

  const bool isMatMulOverlap =  thisInput != nullptr && thisOutput != nullptr && thisWeight != nullptr;
  // if (threadIdx.x == 0) {
  //   printf("isMatMulOverlap %d\n", isMatMulOverlap);
  // }

  #define FINE_GRAINED_OVERLAP 1
  #define COURSE_GRAINED_OVERLAP 2
  #define FIRST_FINE_THEN_COARSE_OVERLAP 3

  #define USE_CUBLAS 1
  #define USE_CUSTOM_MATMUL 2

  #define CUSTOM_INDEXING true

  const int OverlapType = FIRST_FINE_THEN_COARSE_OVERLAP;
  const int KernelType = USE_CUBLAS;

  if (OverlapType == FIRST_FINE_THEN_COARSE_OVERLAP) 
    assert(KernelType == USE_CUBLAS);

  if (KernelType == USE_CUSTOM_MATMUL && isMatMulOverlap) {
    if (threadIdx.x == 0) {
    }

    __syncthreads();
  }

  int MATMUL_M = args->MATMUL_M;
  int MATMUL_N = args->MATMUL_N;
  int MATMUL_K = args->MATMUL_K;

  half* matrixM1 = (half*)thisInput;
  half* matrixM2 = (half*)thisOutput;
  half* matrixM1M2 =  (half*)thisWeight;

  if (isMatMulOverlap) {
    thisOutput = thisWeight;
    thisInput = thisWeight;
    thisWeight = nullptr;
  }
  else if (thisWeight != nullptr) {
    thisOutput = thisWeight;
  }

  if (isMatMulOverlap) {
    ssize_t gridOffset = 0;
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    if (KernelType == USE_CUSTOM_MATMUL) {
      if (OverlapType == FINE_GRAINED_OVERLAP) {
        //First Step 0 Matmul
        if (threadIdx.x == 0) {

        }
      } else if (OverlapType == COURSE_GRAINED_OVERLAP) {
        nelem = nranks*realChunkSize;
        offset = chunkOffset;
        if (threadIdx.x == 0) {
          // printf("chunkOffset %ld nranks*realChunkSize %d realChunkSize %d rank %d\n", chunkOffset, nelem, realChunkSize, comm->rank);
        }
        if (chunkOffset + nelem > size) nelem = size - chunkOffset;
        if (threadIdx.x == 0 and nelem > 0) {

        }
      }
    }
  }

  __syncwarp();
  
  volatile int* fineGrainedOverlapSyncMem = thisSyncGlobalMem;
  volatile int* courseGrainedOverlapSyncMem = thisSyncGlobalMem + args->nChannels;

  ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, FUNC>
    prims(tid, args->nThreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, args->opCount,
    fineGrainedOverlapSyncMem);
  
  curandState randState;
  if (optimizerType == DropoutBiasLayerNorm) {
    curand_init(threadIdx.x, 0, 0, &randState);
    prims.setCurandState(&randState);
    assert(bias != nullptr);
    assert(addTensor != nullptr);
  }

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));
    if (MATMUL_N == 3072 and args->nChannels == 12) {
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T)*3);
    }
    else {
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    }

    ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

    
    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    
    SyncChildKernel;

    //MatMul for step 0
    if (isMatMulOverlap && threadIdx.x == 0) {
      ssize_t gridOffsetStep0 = gridOffset + nranks*loopSize;
      if (gridOffsetStep0 < size) {
        int realChunkSize = min(chunkSize, DIVUP(size-gridOffsetStep0,nranks*args->nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffsetStep0 + bid*nranks*realChunkSize;

        /////////////// begin AllReduce steps ///////////////
        ssize_t offset;
        int nelem;
        int chunk;

        // step 0: push data to next GPU
        chunk = ring->devUserRanks[nranks-1];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset); 
        if (KernelType == USE_CUSTOM_MATMUL) {
          if (OverlapType == COURSE_GRAINED_OVERLAP) {
            nelem = nranks*realChunkSize;
            offset = chunkOffset;
            if (threadIdx.x == 0) {
              // printf("chunkOffset %ld nranks*realChunkSize %d realChunkSize %d rank %d\n", chunkOffset, nelem, realChunkSize, comm->rank);
            }
            if (chunkOffset + nelem > size) nelem = size - chunkOffset;
            if (threadIdx.x == 0 and nelem > 0) {
            }
          }
        }
      }
    }
    __syncwarp();

    // step 0: push data to next GPU
    if (CUSTOM_INDEXING && gridOffset == 0) {
      ssize_t chunkSize2 = DIVUP(size-gridOffset, args->nChannels*nranks*realChunkSize)*realChunkSize;
      chunk = ring->devUserRanks[nranks-1];
      offset = gridOffset + (chunk*args->nChannels+bid) * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      // int origchunk = ring->devUserRanks[nranks-1];
      // ssize_t origoffset = chunkOffset + origchunk * realChunkSize;
      // int orignelem = min(realChunkSize, size-origoffset);
      // if (comm->rank == 0 && threadIdx.x == 0) {
      //   printf("channel %d gridOffset %ld chunk %d nelem %d offset %ld chunkSize2 %ld orignelem %d origoffset %ld realChunkSize %d nranks*loopSize %ld\n",
      //          bid, gridOffset, chunk, nelem, offset, chunkSize2, orignelem, origoffset, realChunkSize, nranks*loopSize);
      // }
    } else {
      chunk = ring->devUserRanks[nranks-1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);
    }
    
    int fineGrainedOverlapSyncMemLastReadVal = -1; //Accessed only in thread 0 of threadblock

    if (KernelType == USE_CUBLAS && threadIdx.x == 0 && thisSyncGlobalMem != nullptr) {
      // printf("Expected Value %ld Current Value %d bid %d rank %d gridOffset %ld size %ld\n", 
          // gridOffset/(nranks*loopSize), *(thisSyncGlobalMem + bid), bid, comm->rank, gridOffset, size);
      if (OverlapType ==  COURSE_GRAINED_OVERLAP || (OverlapType == FIRST_FINE_THEN_COARSE_OVERLAP && gridOffset > 0)) {
        while (*(courseGrainedOverlapSyncMem + bid) < gridOffset/(nranks*loopSize)) {
          //Wait until rows to be used by this iteration has been generated by cuBLAS.
        }
      } else if (OverlapType == FINE_GRAINED_OVERLAP || (OverlapType == FIRST_FINE_THEN_COARSE_OVERLAP && gridOffset == 0)) {
        int j = 1;
        
        while ((fineGrainedOverlapSyncMemLastReadVal = *(fineGrainedOverlapSyncMem + bid)) < j) {
          //Wait until rows to be used by this iteration has been generated by cuBLAS.
        }
      }
    }

    __syncwarp();
    __syncthreads();

    if (KernelType == USE_CUSTOM_MATMUL && OverlapType == FINE_GRAINED_OVERLAP && isMatMulOverlap && nranks > 2) {
      int j = 2;
      ssize_t offsetStep2;
      int nelemStep2;
      int chunkStep2;

      //Matmul for Step 2
      chunkStep2 = ring->devUserRanks[nranks-j];
      offsetStep2 = chunkOffset + chunkStep2 * realChunkSize;
      nelemStep2 = min(realChunkSize, size-offsetStep2);

      if (threadIdx.x == 0) {
      }
    }

    __syncwarp();

    prims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      if (CUSTOM_INDEXING && gridOffset == 0) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*args->nChannels+bid) * realChunkSize;
        nelem = min(realChunkSize, size-offset);
      } else {
        //M x N = 16384 * 4096  = 64 Mega elements. 
        //Per Channels = 64 / 16 = 4 Mega Elements
        //Per GPU = 1 Mega Element
        //Number of Rows = 1 Mega / 4096 = 256 Rows.
        //Ideal way = 256 * # of Channel Rows = 256 * 16 = 4096 rows
        chunk = ring->devUserRanks[nranks-j]; 
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);
      }

      SyncChildKernel;
      
      
      if ((OverlapType == FINE_GRAINED_OVERLAP || (OverlapType == FIRST_FINE_THEN_COARSE_OVERLAP && gridOffset == 0)) && isMatMulOverlap && threadIdx.x == 0) {
        ssize_t offsetStepJPlus1;
        int nelemStepJPlus1;
        int chunkStepJPlus1;

        //Matmul for Step j + 1
        chunkStepJPlus1 = ring->devUserRanks[nranks-(j + 1)];
        offsetStepJPlus1 = chunkOffset + chunkStepJPlus1 * realChunkSize;
        nelemStepJPlus1 = min(realChunkSize, size-offsetStepJPlus1);

        if (KernelType == USE_CUSTOM_MATMUL && j + 1 < nranks) {
        } else if (false && KernelType == USE_CUBLAS) {
          if (fineGrainedOverlapSyncMemLastReadVal < j) {
            while ((fineGrainedOverlapSyncMemLastReadVal = *(fineGrainedOverlapSyncMem + bid)) < j) {
              //Wait until rows to be used by this iteration has been generated by cuBLAS.
            }
          }
        }
      }
      __syncwarp();
      if (isMatMulOverlap && gridOffset == 0)
        prims.recvReduceSendOverlappedWithMatmul(thisInput+offset, nelem, j);
      else {
        if (KernelType == USE_CUBLAS) {
          __syncwarp();
          __syncthreads();
        }
        prims.recvReduceSend(thisInput+offset, nelem);
      }
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    if (CUSTOM_INDEXING && gridOffset == 0) {
      chunk = ring->devUserRanks[0];
      offset = gridOffset + (chunk*args->nChannels+bid) * realChunkSize;
      nelem = min(realChunkSize, size-offset);
    } else {
      chunk = ring->devUserRanks[0];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);
    }
    // if (threadIdx.x == 0 && bid == 0) {
    //   printf("step %d rank %d nelem %d offset %ld channel %d\n", nranks, comm->rank, nelem, offset, bid);
    // }

    if ((OverlapType == FINE_GRAINED_OVERLAP || (OverlapType == FIRST_FINE_THEN_COARSE_OVERLAP && gridOffset == 0)) && isMatMulOverlap && threadIdx.x == 0) {
      if (KernelType == USE_CUSTOM_MATMUL) {
      } else if (KernelType == USE_CUBLAS) {
        if (fineGrainedOverlapSyncMemLastReadVal < nranks) {
          while ((fineGrainedOverlapSyncMemLastReadVal = *(fineGrainedOverlapSyncMem + bid)) < nranks) {
            //Wait until rows to be used by this iteration has been generated by cuBLAS.
          }
        }
      }
    }

    __syncwarp();
    __syncthreads();

    if (optimizerType == OptimizerType::DropoutBiasLayerNorm) {
      prims.directRecvReduceCopySendDropoutBiasLayernorm(thisInput+offset, thisOutput+offset, addTensor, bias, biasSize, offset, nelem);
    } else if (thisWeight == nullptr) {
      prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);
    } else if (doAdam) {
      prims.directRecvReduceCopySendAdam(thisInput+offset, thisWeight+offset, thisFirstMoment+offset, thisSecondMoment+offset, 
                                         offset, nelem, alpha, beta1, beta2, epoch);
    } else {
      prims.directRecvReduceCopySendWeight(thisInput+offset, thisWeight+offset, offset, nelem, alpha);
    }

    //MatMul for step 0
    if (isMatMulOverlap && threadIdx.x == 0) {
      ssize_t gridOffsetStep0 = gridOffset + nranks*loopSize;
      if (gridOffsetStep0 < size) {
        int realChunkSize = min(chunkSize, DIVUP(size-gridOffsetStep0,nranks*args->nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffsetStep0 + bid*nranks*realChunkSize;

        /////////////// begin AllReduce steps ///////////////
        ssize_t offset;
        int nelem;
        int chunk;

        // step 0: push data to next GPU
        chunk = ring->devUserRanks[nranks-1];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset); 

        if (KernelType == USE_CUSTOM_MATMUL && OverlapType == FINE_GRAINED_OVERLAP) {
        }
      }
    }
      __syncwarp();

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      if (CUSTOM_INDEXING && gridOffset == 0) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*args->nChannels+bid) * realChunkSize;
        nelem = min(realChunkSize, size-offset);
      } else {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);
      }

      if (thisWeight == nullptr) {
        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      } else {
        prims.directRecvCopySend(thisWeight+offset, offset, nelem);
      }
    }

    // Make final copy from buffer to dest.
    if (CUSTOM_INDEXING && gridOffset == 0) {
        chunk = ring->devUserRanks[1];
        offset = gridOffset + (chunk*args->nChannels+bid) * realChunkSize;
        nelem = min(realChunkSize, size-offset);
    } else {
      chunk = ring->devUserRanks[1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);
    }

    // Final wait/copy.
    if (thisWeight == nullptr) {
      prims.directRecv(thisOutput+offset, offset, nelem);
    } else {
      prims.directRecv(thisWeight+offset, offset, nelem);
    }
  }
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceTreeKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads-WARP_SIZE;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const ssize_t size = args->N;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  int chunkSize = args->lastChunkSize;
  const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;
  
  if (loopSize > size) {
    chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  
  do {
    struct ncclTree* tree = &channel->treeUp;
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclPrimitives<UNROLL/2, 1, 1, T, NCCL_MAX_TREE_ARITY, 1, FUNC> prims(tid, args->nThreads, tree->down, &tree->up, NULL, stepSize, channel, comm, args->opCount, nullptr);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.send(thisInput+offset, nelem);
      } else {
        prims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  } while(0);

  do {
    struct ncclTree* tree = &channel->treeDn;
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclPrimitives<UNROLL/2, 1, 1, T, 1, NCCL_MAX_TREE_ARITY, FUNC> prims(tid, args->nThreads, &tree->up, tree->down, NULL, stepSize, channel, comm, args->opCount, nullptr);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.recv(thisOutput+offset, nelem);
      } else {
        prims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  } while(0);
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceCollNetKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads-WARP_SIZE;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const ssize_t size = args->N;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  int chunkSize = args->lastChunkSize;
  const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;
  
  if (loopSize > size) {
    chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  if (blockIdx.x < args->nChannels) { // first half of the channels do reduce
    struct ncclTree* tree = &channel->collTreeUp;
    ncclPrimitives<UNROLL, 1, 1, T, 1, 1, FUNC> prims(tid, args->nThreads, tree->down, &tree->up, NULL, stepSize, channel, comm, args->opCount, nullptr);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.send(thisInput+offset, nelem);
      } else {
        prims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  }

  if (blockIdx.x >= args->nChannels) { // second half of the channels do broadcast
    struct ncclTree* tree = &channel->collTreeDn;
    ncclPrimitives<UNROLL, 1, 1, T, 1, 1, FUNC> prims(tid, args->nThreads, &tree->up, tree->down, NULL, stepSize, channel, comm, args->opCount, nullptr);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.recv(thisOutput+offset, nelem);
      } else {
        prims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceRingLLKernel(struct CollectiveArgs* args) {
  // if (threadIdx.x + blockDim.x * blockIdx.x == 0) 
  //   printf("RingLLKernel\n");
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  
  ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, channel, comm, args->opCount);
  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads * (sizeof(uint64_t)) / sizeof(T);

  const ssize_t loopSize = args->nChannels*nranks*chunkSize;
  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;

  //Weights and Grads are contiguous
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  T * __restrict__ thisWeight = (T*)args->ThisWeight;
  const T * __restrict__ thisAlpha = (T*)args->ThisAlpha;
  const T alpha = (thisAlpha == NULL) ? (T)0.0 : *thisAlpha;
  const T beta1 = (args->ThisBeta1 == NULL) ? (T)0.0 : *(T*)args->ThisBeta1;
  const T beta2 = (args->ThisBeta1 == NULL) ? (T)0.0 : *(T*)args->ThisBeta2;
  T* __restrict__ thisFirstMoment = (T*)args->ThisFirstMoment;
  T* __restrict__ thisSecondMoment = (T*)args->ThisSecondMoment;
  const int epoch = args->epoch;
  const bool doAdam = (thisFirstMoment != NULL && thisSecondMoment != NULL);
  
  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    chunkSize = min(DIVUP(size-gridOffset, args->nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    if (thisWeight == nullptr) {
      LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
    } else if (doAdam) {
      LLprims.recvReduceUpdateandSendWeightInAdam(thisInput+offset, thisOutput+offset, thisWeight+offset, thisWeight+offset, 
                                                  thisFirstMoment+offset, thisSecondMoment+offset, alpha, beta1, beta2, epoch, nelem, 0);
    } else {
      //SGD
      //Reduce the gradients, then update and send weights
      LLprims.recvReduceUpdateandSendWeight(thisInput+offset, thisOutput+offset, thisWeight+offset, thisWeight+offset, alpha, nelem, 0);
    }

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      if (thisWeight == nullptr) {
        LLprims.recvCopySend(thisOutput+offset, nelem);
      } else {
        //Receive and send the updated weights
        //Same for SGD and Adam
        LLprims.recvCopySend(thisWeight+offset, nelem);
      }
    }
    
    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    if (thisWeight == nullptr) {
      LLprims.recv(thisOutput+offset, nelem);
    } else {
      //Receive the updated weights
      //Same for SGD and Adam
      LLprims.recv(thisWeight+offset, nelem);
    }
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceTreeLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const ssize_t size = args->N;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;
  
  if (loopSize > size) {
    chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  do {
    struct ncclTree* tree = &channel->treeUp;
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclLLPrimitives<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreads, tree->down, &tree->up, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  } while(0);

  do {
    struct ncclTree* tree = &channel->treeDn;
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclLLPrimitives<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, &tree->up, tree->down, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.recv(thisOutput+offset, nelem);
      } else {
        LLprims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  } while(0);
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceCollNetLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const ssize_t size = args->N;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;
  
  if (loopSize > size) {
    chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  if (blockIdx.x < args->nChannels) { // first half of the channels do reduce
    struct ncclTree* tree = &channel->collTreeUp;
    ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, tree->down, &tree->up, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  }

  if (blockIdx.x >= args->nChannels) { // second half of the channels do broadcast
    struct ncclTree* tree = &channel->collTreeDn;
    ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &tree->up, tree->down, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.recv(thisOutput+offset, nelem);
      } else {
        LLprims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  }
}

#include "prims_ll128.h"
template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceRingLL128Kernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  
  ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, channel, comm, args->opCount);

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = (NCCL_LL128_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));
  // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
  const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;

  const ssize_t loopSize = args->nChannels*nranks*chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    chunkSize = min(DIVUP(size-gridOffset, args->nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv(thisOutput+offset, nelem);
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceTreeLL128Kernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclTree* treeUp = &channel->treeUp;
  struct ncclTree* treeDn = &channel->treeDn;
  const ssize_t size = args->N;
  ssize_t chunkSize = args->lastChunkSize;
  const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/8;
  const ssize_t loopSize = args->nChannels*chunkSize;
  int nthreadsSplit = NCCL_LL128_SPLIT(nthreads);
  
  if (loopSize > size) {
    chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  if (treeUp->up == -1) {
    // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
    ncclLL128Primitives<T, FUNC, NCCL_MAX_TREE_ARITY, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, treeUp->down, treeDn->down, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
    }
  } else {
    if (tid < nthreadsSplit) {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      ncclLL128Primitives<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreadsSplit, treeUp->down, &treeUp->up, channel, comm, args->opCount);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (treeUp->down[0] == -1) {
          LLprims.send(thisInput+offset, nelem);
        } else {
          LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    } else {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      ncclLL128Primitives<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid-nthreadsSplit, nthreads-nthreadsSplit, &treeDn->up, treeDn->down, channel, comm, args->opCount);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (treeDn->down[0] == -1) {
          LLprims.recv(thisOutput+offset, nelem);
        } else {
          LLprims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    }
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceCollNetLL128Kernel(struct CollectiveArgs* args) { }
