#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <vector>
#include <mpi.h>
#include <algorithm>
#include <cuda_fp16.h>

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

#define NCCL_STEPS 8
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)

const int maxBuffSize = 1024;

#define CURANDCHECK(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);            \
	assert(false);}} while(0)

#define CUBLASCHECK(cmd) do {                       \
	cublasStatus_t e = cmd;                           \
	if (e != CUBLAS_STATUS_SUCCESS) {                 \
		printf("Failed: CUBLAS error %s: %d '%d'\n",    \
				__FILE__, __LINE__, cmd);                \
		assert(false);                                  \
	}                                                 \
} while(0)                                          \

#define CUDACHECK(cmd) do {                         \
	cudaError_t e = cmd;                              \
	if( e != cudaSuccess ) {                          \
		printf("Failed: Cuda error %s:%d '%s'\n",             \
				__FILE__,__LINE__,cudaGetErrorString(e));   \
		exit(EXIT_FAILURE);                             \
	}                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
	ncclResult_t r = cmd;                             \
	if (r!= ncclSuccess) {                            \
		printf("Failed, NCCL error %s:%d '%s'\n",             \
				__FILE__,__LINE__,ncclGetErrorString(r));   \
		exit(EXIT_FAILURE);                             \
	}                                                 \
} while(0)


template<class T>
bool eq_float(T f1, T f2)
{
  if (f1 == (T)0.0 && f2 == (T)0.0)
    return true;
  return (fabs((f1-f2)/std::max(f1, f2)) <= 1e-4);
}

template<class T>
void memset_value(T*f, T v, size_t nelems) 
{
  T* h_buff = (T*)malloc(sizeof(T)*nelems);

  for (uint64_t i = 0; i < nelems; i++) {
    h_buff[i] = v;
  }

  CUDACHECK(cudaMemcpy(f, h_buff, sizeof(T)*nelems, cudaMemcpyHostToDevice));
  free(h_buff);
}

template<class T>
bool check_epoch_results_adam(int rank, int algo, int buffIndex, int epoch,
                        const uint64_t num_weight_elems,
                        T* h_minibatch_gradients, 
                        T* h_weights, T* h_new_weights, 
                        T* cpu_moment,
                        T* cpu_second_moment,
                        T beta1, T beta2, T stepsize, T epsilon)	
{
  bool passed = true;
//   T **h_minibatch_gradients = (T**)malloc(nDev * sizeof(T*));
  const size_t grad_array_size = num_weight_elems*sizeof(T);

//   //Check AllReduced
//   for (int dev = 0; dev < nDev; dev++) {
//     CUDACHECK(cudaSetDevice(dev));
//     h_minibatch_gradients[dev] = (T*)malloc(num_weight_elems*sizeof(T));
//     CUDACHECK(cudaMemcpy(h_minibatch_gradients[dev], d_minibatch_gradients[dev], 
//                          grad_array_size, cudaMemcpyDeviceToHost));
//   }

  T *h_reduced_grad = (T*)malloc(grad_array_size);

  MPI_Allreduce(h_minibatch_gradients, h_reduced_grad, num_weight_elems, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

   //Check Weight Update


  for (uint64_t i = 0; i < num_weight_elems; i++) {
    T m, v;
    T old_m = cpu_moment[i];

    m = beta1 * old_m + (1-beta1) * h_reduced_grad[i];

    T old_v = cpu_second_moment[i];
    v = beta2 * old_v + (1-beta2) * h_reduced_grad[i]*h_reduced_grad[i];
    
    T m_ = m/(1 - pow(beta1, epoch + 1));
    T v_ = v/(1 - pow(beta2, epoch + 1));
    T x = stepsize * m_ / (sqrt(v_) + epsilon);
    T new_weight = __half2float(__float2half(h_weights[i] + x));
	if (epoch <= 1 and i == 0 and buffIndex == 0) {
		printf("old_m %f storing m %f\n", old_m, m);
	}
    if (!eq_float(new_weight, h_new_weights[i])) {
      //Lets take a look at the last device only.
      printf("rank %d BuffIndex %d Epoch %d Mismatch in h_new_weights at [%ld]: ref '%f' computed '%f'\n", rank, buffIndex, epoch, i, new_weight, h_new_weights[i]);
      printf("h_weights[i] %f x %f != incorrect %f, pow %f, m_ %f, v_ %f, beta1 %f, beta2 %f, old_m %f, old_v %f h_reduced_grad[i] %f\n", h_weights[i], x, h_new_weights[i], pow(beta1, epoch + 1), m_, v_, beta1, beta2, m, v, h_reduced_grad[i]);
      // printf("h_weights[%ld] = %f , h_reduced_gradients[%ld] = %f, new_m = %f, new_v = %f, m_ = %f, v_ = %f, old_m = %f, old_v = %f d_new_m = %f\n", i, h_weights[dev][i], i, h_reduced_grad[i], m, v, m_, v_, old_m, old_v, h_moments[dev][i]);
      printf("\n");
      passed = false;
      break;
    }

    cpu_second_moment[i] = v;
    cpu_moment[i] = m;

    if (!passed)
      break;
  }

  //Correct these to free
// free(h_minibatch_gradients[i]);
// free(h_weights[i]);
// free(h_new_weights[i]);

//   free(h_new_weights);
//   free(h_reduced_grad);
//   free(h_weights);
//   free(h_minibatch_gradients);

  return passed;
}


void halfToFloatArray(float* f, half* h, size_t num) 
{
  for (size_t i = 0; i < num; i++) {
    f[i] = __half2float(h[i]);
  }
}


void cudaMemcpyHalfDevice2FloatHost(float* hostFloatArray, half* deviceHalfArray, size_t nelems)
{
  half* tmp = new half[nelems];
  CUDACHECK(cudaMemcpy(tmp, deviceHalfArray, nelems*sizeof(half), cudaMemcpyDeviceToHost));

  halfToFloatArray(hostFloatArray, tmp, nelems);

  delete tmp;
}

float run_scattered_adam_mp(uint64_t nbuffs, const size_t* h_buffSizes, ncclDataType_t datatype, bool check_results = true)
{
	typedef half T;
	const int epochs = 20;
	int comm_size, rank;
  	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  	ncclComm_t comm;
  	CUDACHECK(cudaSetDevice(rank % 16));

	//initializing NCCL
  	ncclUniqueId id;
	if (rank == 0)
	  ncclGetUniqueId(&id);
  	MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  	ncclCommInitRank(&comm, comm_size, id, rank);

	T** gradients;
	T** weights;
	float** floatWeights;
	float** firstMoment;
	float** secondMoment;
	float* alphas;
	float* beta1s;
	float* beta2s;
	float* unscaleParameters;
	int* numOverflows;

	T* realGpuGrads;
	T* realGpuWeights;
	float* realFloatGpuWeights;
	float* realGpuFirstMoment;
	float* realGpuSecondMoment;


	const float beta1 = 0.5f;
	const float beta2 = 0.5f;
	const float alpha = 1.0f;
	const float unscaleParameter = 1.0f;

	cudaStream_t s;
	CUDACHECK(cudaStreamCreate(&s));
	size_t* buffSizes;

MPI_Barrier(MPI_COMM_WORLD);
	std::vector<size_t> buffSizesAndLayerId;
	size_t totalSize = 0;
	for (size_t i = 0; i < nbuffs; i++) {
		int nSmallBuffersForThisBuffer = h_buffSizes[i] / maxBuffSize;
		if ((h_buffSizes[i] % maxBuffSize) != 0){
			nSmallBuffersForThisBuffer++;
		}
		for (int j = 0; j < nSmallBuffersForThisBuffer; j++){
			buffSizesAndLayerId.push_back((std::min((size_t) (h_buffSizes[i] - j*maxBuffSize), (size_t) maxBuffSize) % (1<<10)) + (j << 10) + ((size_t)nSmallBuffersForThisBuffer << 35));
			// if (rank == 0) {
      //   printf("i: %ld j: %ld smallBuffSize: %ld numSmallBufs: %ld\n", i, j, std::min((size_t) (h_buffSizes[i] - j*maxBuffSize), (size_t) maxBuffSize)  % (1 << 10), nSmallBuffersForThisBuffer);
			// }
		}
		totalSize += nSmallBuffersForThisBuffer*maxBuffSize;
		// buffSizesAndLayerId[i] = (h_buffSizes[i] % (1<<10)) + (i << 10) + (nbuffs << 20);
	}
	if (rank == 0)
	printf("\n");
	MPI_Barrier(MPI_COMM_WORLD);
	std::cout << "totalSize " << totalSize << " buffSizesAndLayerId " << buffSizesAndLayerId.size() << std::endl;
	float** cpuMomentum = (float**)malloc(nbuffs * sizeof(float*));
	float** cpuVelocity = (float**)malloc(nbuffs * sizeof(float*));
	T** cpuGrads = (T**)malloc(nbuffs * sizeof(T*));
	T** cpuWeight = (T**)malloc(nbuffs * sizeof(T*));
	float** cpuFloatWeight = (float**)malloc(nbuffs * sizeof(float*));


	for (size_t i = 0; i < nbuffs; i++) {
		//printf("h_buffSizes[%d] %ld sizeof(T) %ld\n", i, h_buffSizes[i], sizeof(T));
		cpuVelocity[i] = (float*)malloc(h_buffSizes[i]*sizeof(float));
		cpuMomentum[i] = (float*)malloc(h_buffSizes[i]*sizeof(float));
		cpuGrads[i] = (T*)malloc(h_buffSizes[i]*sizeof(T));
		cpuWeight[i] = (T*)malloc(h_buffSizes[i]*sizeof(T));
		cpuFloatWeight[i] = (float*)malloc(h_buffSizes[i]*sizeof(float));
		for (size_t index = 0; index < h_buffSizes[i]; index++){
			cpuVelocity[i][index] = 0.f;
			cpuMomentum[i][index] = 0.f;
			float w = (float)(i%2 + 1)*(index%20+1);
			cpuGrads[i][index] = __float2half(w);
			cpuWeight[i][index] = __float2half(w);
			cpuFloatWeight[i][index] = w;
		}
	}

	int smallNbuffer = buffSizesAndLayerId.size();
	CUDACHECK(cudaMalloc(&buffSizes, smallNbuffer * sizeof(size_t)));
	CUDACHECK(cudaMemcpy(buffSizes, buffSizesAndLayerId.data(), smallNbuffer * sizeof(size_t), cudaMemcpyHostToDevice));
	CUDACHECK(cudaMalloc(&weights, smallNbuffer * sizeof(T*)));
	CUDACHECK(cudaMalloc(&gradients, smallNbuffer * sizeof(T*)));
	CUDACHECK(cudaMalloc(&floatWeights, smallNbuffer * sizeof(float*)));

	CUDACHECK(cudaMalloc(&alphas, sizeof(float)));
	CUDACHECK(cudaMalloc(&beta1s, sizeof(float)));
	CUDACHECK(cudaMalloc(&beta2s, sizeof(float)));
	CUDACHECK(cudaMalloc(&unscaleParameters, sizeof(float)));
	CUDACHECK(cudaMalloc(&numOverflows, sizeof(int)));

	CUDACHECK(cudaMemcpy(alphas, &alpha, sizeof(float), cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(beta1s, &beta1, sizeof(float), cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(beta2s, &beta2, sizeof(float), cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(unscaleParameters, &unscaleParameter, sizeof(float), cudaMemcpyHostToDevice));
	
	T** tmpGrads = (T**)malloc(smallNbuffer * sizeof(T**));
	T** tmpWeights = (T**)malloc(smallNbuffer * sizeof(T**));
	float** tmpFloatWeights = (float**)malloc(smallNbuffer * sizeof(float**));
	float** tmpFirstMoment = (float**)malloc(smallNbuffer * sizeof(float**));
	float** tmpSecondMoment = (float**)malloc(smallNbuffer * sizeof(float**));
	float* singleBufferSecondMoment = nullptr;
	float* singleBufferFirstMoment = nullptr;
	
	size_t mSizeForSimple = 0;
	{
		assert (atoi(getenv ("NCCL_MIN_NCHANNELS")) == atoi(getenv ("NCCL_MAX_NCHANNELS")));
		int nChannels = atoi(getenv ("NCCL_MIN_NCHANNELS"));
		int nThreads = atoi(getenv("NCCL_NTHREADS"));
		int channelBuffSize = atoi(getenv("NCCL_BUFFSIZE"));
		const int stepSize = channelBuffSize / (sizeof(T)*NCCL_STEPS);
		const size_t chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
		size_t maxPartSizeForSimple = std::min(chunkSize, DIVUP(totalSize,comm_size*nChannels));
		ALIGN_SIZE(maxPartSizeForSimple, nThreads*sizeof(uint64_t)/sizeof(T));
		const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
		const ssize_t numParts = DIVUP(totalSize, comm_size*loopSize);
		std::cout << "numParts " << numParts <<" maxPartSizeForSimple " << maxPartSizeForSimple << " chunkSize " << chunkSize << std::endl;
		mSizeForSimple = maxPartSizeForSimple*nChannels*numParts;
		std::cout << "mSizeForSimple " << mSizeForSimple << std::endl;
		CUDACHECK(cudaMalloc(&singleBufferSecondMoment, mSizeForSimple * sizeof(float)));
		CUDACHECK(cudaMalloc(&singleBufferFirstMoment, mSizeForSimple * sizeof(float)));
		CUDACHECK(cudaMemset(singleBufferSecondMoment, 0, mSizeForSimple * sizeof(float)));
		CUDACHECK(cudaMemset(singleBufferFirstMoment, 0, mSizeForSimple * sizeof(float)));
	}

	CUDACHECK(cudaMalloc(&realGpuGrads, totalSize * sizeof(T)));
	CUDACHECK(cudaMalloc(&realGpuWeights, totalSize * sizeof(T)));
	CUDACHECK(cudaMalloc(&realFloatGpuWeights, totalSize * sizeof(float)));
	
	int accSize = 0;
	int curBuffIndex = 0;
	for (size_t i = 0; i < nbuffs; i++){
		int nSmallBuffersForThisBuffer = h_buffSizes[i] / maxBuffSize;
		if ((h_buffSizes[i] % maxBuffSize) != 0){
			nSmallBuffersForThisBuffer++;
		}
		CUDACHECK(cudaMemcpy(&realGpuGrads[accSize], cpuGrads[i], h_buffSizes[i] * sizeof(T), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(&realGpuWeights[accSize], cpuWeight[i], h_buffSizes[i] * sizeof(T), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(&realFloatGpuWeights[accSize], cpuFloatWeight[i], h_buffSizes[i] * sizeof(float), cudaMemcpyHostToDevice));

		// CUDACHECK(cudaMemcpy(&realGpuFirstMoment[accSize], cpuMomentum[i], h_buffSizes[i] * sizeof(T), cudaMemcpyHostToDevice));
		// CUDACHECK(cudaMemcpy(&realGpuSecondMoment[accSize], cpuVelocity[i], h_buffSizes[i] * sizeof(T), cudaMemcpyHostToDevice));
		for (int j = 0; j < nSmallBuffersForThisBuffer; j++){
			tmpGrads[curBuffIndex] = &realGpuGrads[accSize + j*maxBuffSize];
			tmpWeights[curBuffIndex] = &realGpuWeights[accSize + j*maxBuffSize];
			tmpFloatWeights[curBuffIndex] = &realFloatGpuWeights[accSize + j*maxBuffSize];
			//tmpFirstMoment[curBuffIndex] = &realGpuFirstMoment[accSize + j*maxBuffSize];
			//tmpSecondMoment[curBuffIndex] = &realGpuSecondMoment[accSize + j*maxBuffSize];
			curBuffIndex++;
		}
		
		accSize += nSmallBuffersForThisBuffer * maxBuffSize;
	}
	CUDACHECK(cudaMemcpy(gradients, tmpGrads, sizeof(T*) * smallNbuffer, cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(weights, tmpWeights, sizeof(T*) * smallNbuffer, cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(floatWeights, tmpFloatWeights, sizeof(float*) * smallNbuffer, cudaMemcpyHostToDevice));

	float** tmpOldWeights = new float*[smallNbuffer];
	float** tmpOldFloatWeights = new float*[smallNbuffer];
	float** tmpOldGrads = new float*[smallNbuffer];
	float** tmpOldFirstMomentum = new float*[smallNbuffer];
	float** tmpOldSecondMomentum = new float*[smallNbuffer];
	float** tmpNewWeights = new float*[smallNbuffer];

	for (size_t buff = 0; buff < smallNbuffer; buff++) {
		int thisBufferSize = ((buffSizesAndLayerId[buff] % maxBuffSize) == 0) ? maxBuffSize : (buffSizesAndLayerId[buff] % maxBuffSize);

		tmpOldWeights[buff] = new float[thisBufferSize];
		tmpOldFloatWeights[buff] = new float[thisBufferSize];
		tmpOldGrads[buff] = new float[thisBufferSize];
		tmpOldFirstMomentum[buff] = new float[thisBufferSize];
		tmpOldSecondMomentum[buff] = new float[thisBufferSize];
		tmpNewWeights[buff] = new float[thisBufferSize];
		memset(tmpOldFirstMomentum[buff], 0, thisBufferSize*sizeof(float));
		memset(tmpOldSecondMomentum[buff], 0, thisBufferSize*sizeof(float));
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	float totalTime = 0.0;

	for (int iter = 0; iter < epochs + 5; iter++) {
    	// printf("iter %d\n", iter);
		if (check_results and iter < 5) {
			for (size_t buff = 0; buff < smallNbuffer; buff++) {
				int thisBufferSize = ((buffSizesAndLayerId[buff] % maxBuffSize) == 0) ? maxBuffSize : (buffSizesAndLayerId[buff] % maxBuffSize);
				cudaMemcpy(tmpOldFloatWeights[buff], tmpFloatWeights[buff], sizeof(float)*thisBufferSize, cudaMemcpyDeviceToHost);
				cudaMemcpyHalfDevice2FloatHost(tmpOldGrads[buff], tmpGrads[buff], thisBufferSize);
				cudaMemcpyHalfDevice2FloatHost(tmpOldWeights[buff], tmpWeights[buff], thisBufferSize);
        		//cudaMemcpy(tmpOldFirstMomentum[buff], tmpFirstMoment[buff], sizeof(T)*thisBufferSize, cudaMemcpyDeviceToHost);
        		//cudaMemcpy(tmpOldSecondMomentum[buff], tmpSecondMoment[buff], sizeof(T)*thisBufferSize, cudaMemcpyDeviceToHost);
			}
		}
		cudaEvent_t start, stop;
		CUDACHECK(cudaEventCreate(&start));
		CUDACHECK(cudaEventCreate(&stop));
		CUDACHECK(cudaEventRecord(start,0));
		NCCLCHECK(ncclAllReduceScatteredAdamMP((const void**)gradients, (void**)weights, floatWeights, (void**)singleBufferFirstMoment, 
											   (void**)singleBufferSecondMoment, smallNbuffer, buffSizes,
											   totalSize, (void*) alphas, (void*) beta1s, (void*)beta2s, (void*)unscaleParameters, numOverflows, iter, datatype, 
											   ncclSum, comm, s));
		CUDACHECK(cudaEventRecord(stop,0));
		CUDACHECK(cudaEventSynchronize(stop));
		float elapsedTime;
		CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
		if (iter >= 5) 
			totalTime += elapsedTime;

		if (check_results and iter < 5) {
			printf("checking results\n");
			bool passed = true;		
			for (size_t buff = 0; buff < smallNbuffer; buff++) {
				int thisBufferSize = ((buffSizesAndLayerId[buff] % maxBuffSize) == 0) ? maxBuffSize : (buffSizesAndLayerId[buff] % maxBuffSize);
				cudaMemcpyHalfDevice2FloatHost(tmpNewWeights[buff], tmpWeights[buff], thisBufferSize);	
				passed = check_epoch_results_adam<float>(rank, 2, buff, iter, thisBufferSize, tmpOldGrads[buff], tmpOldWeights[buff], tmpNewWeights[buff], tmpOldFirstMomentum[buff], tmpOldSecondMomentum[buff], beta1, beta2, alpha, (float)1e-6);
				if (!passed)
					break;
			}
			if (passed)
				printf("Results checks out!\n");
			else {
				printf("Correctness failed!\n");
				abort();
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		printf("{SZ: %ld, Epochs: %d, Time: %f}\n", totalSize, epochs, totalTime);
	}

	return totalTime/epochs;
}

template<class T>
bool check_epoch_results_lamb(int rank, int algo, int buffIndex, int epoch,
                        const uint64_t num_weight_elems,
                        T* h_minibatch_gradients, 
                        T* h_weights, T* h_new_weights, 
                        T* cpu_moment,
                        T* cpu_second_moment,
                        T beta1, T beta2, T stepsize, T epsilon)	
{
  bool passed = true;
//   T **h_minibatch_gradients = (T**)malloc(nDev * sizeof(T*));
  const size_t grad_array_size = num_weight_elems*sizeof(T);

//   //Check AllReduced
//   for (int dev = 0; dev < nDev; dev++) {
//     CUDACHECK(cudaSetDevice(dev));
//     h_minibatch_gradients[dev] = (T*)malloc(num_weight_elems*sizeof(T));
//     CUDACHECK(cudaMemcpy(h_minibatch_gradients[dev], d_minibatch_gradients[dev], 
//                          grad_array_size, cudaMemcpyDeviceToHost));
//   }

  T *h_reduced_grad = (T*)malloc(grad_array_size);

  MPI_Allreduce(h_minibatch_gradients, h_reduced_grad, num_weight_elems, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

   //Check Weight Update

	double rNorm = 0.0f, wNorm = 0.0f;
  for (uint64_t i = 0; i < num_weight_elems; i++) {
    T m, v;
    T old_m = cpu_moment[i];
    m = beta1 * old_m + (1-beta1) * h_reduced_grad[i];

    T old_v = cpu_second_moment[i];
    v = beta2 * old_v + (1-beta2) * h_reduced_grad[i]*h_reduced_grad[i];
    
    T m_ = m/(1 - pow(beta1, epoch + 1));
    T v_ = v/(1 - pow(beta2, epoch + 1));
    T r = m_ / (sqrt(v_) + epsilon) + 1.0f * h_weights[i];

		rNorm += (r*r)/num_weight_elems;
		wNorm += (h_weights[i] * h_weights[i]);
		// if (epoch == 1 && buffIndex == 115 && !eq_float(h_weights[i],0.5f)) {
		// 	printf("at %ld h_weights[i] %f\n", i, h_weights[i]);
		// }
    cpu_second_moment[i] = v;
    cpu_moment[i] = m;
  }

	rNorm = sqrt(rNorm);
	wNorm = sqrt(wNorm/num_weight_elems);

  for (uint64_t i = 0; i < num_weight_elems; i++) {
		T m = cpu_moment[i];
		T v = cpu_second_moment[i];
		T m_ = m/(1 - pow(beta1, epoch + 1));
    T v_ = v/(1 - pow(beta2, epoch + 1));
    T r = m_ / (sqrt(v_) + epsilon) + 1.0f * h_weights[i];
		T scale = ((wNorm > 0) ? (rNorm > 0 ? wNorm/rNorm : 1.0f) : 1.0f)/rNorm;
    T x = stepsize * scale * r;
    T new_weight = h_weights[i] - x;

    if (!eq_float(new_weight, h_new_weights[i])) {
      //Lets take a look at the last device only.
      printf("rank %d BuffIndex %d Epoch %d Mismatch in h_new_weights at [%ld]: ref '%f' computed '%f'\n", rank, buffIndex, epoch, i, new_weight, h_new_weights[i]);
      printf("h_weights[i] %f x %f != incorrect %f, pow %f, m_ %f, v_ %f, beta1 %f, beta2 %f, old_m %f, old_v %f h_reduced_grad[i] %f rNorm %f wNorm %f num_weight_elems %d r %f scale %f\n", h_weights[i], x, h_new_weights[i], pow(beta1, epoch + 1), m_, v_, beta1, beta2, m, v, h_reduced_grad[i], rNorm, wNorm, num_weight_elems, r, scale);
      // printf("h_weights[%ld] = %f , h_reduced_gradients[%ld] = %f, new_m = %f, new_v = %f, m_ = %f, v_ = %f, old_m = %f, old_v = %f d_new_m = %f\n", i, h_weights[dev][i], i, h_reduced_grad[i], m, v, m_, v_, old_m, old_v, h_moments[dev][i]);
      printf("\n");
      passed = false;
      break;
    }

    if (!passed)
      break;
  }

  //Correct these to free
// free(h_minibatch_gradients[i]);
// free(h_weights[i]);
// free(h_new_weights[i]);

//   free(h_new_weights);
//   free(h_reduced_grad);
//   free(h_weights);
//   free(h_minibatch_gradients);

  return passed;
}

float run_scattered_lamb(uint64_t nbuffs, const size_t* h_buffSizes, ncclDataType_t datatype, bool check_results = true)
{
	typedef half T;
	const int epochs = 20;
	int comm_size, rank;
  	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  	ncclComm_t comm;
  	CUDACHECK(cudaSetDevice(rank % 16));

	//initializing NCCL
  	ncclUniqueId id;
	if (rank == 0)
	  ncclGetUniqueId(&id);
  	MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  	ncclCommInitRank(&comm, comm_size, id, rank);

	T** gradients;
	T** weights;
	float** floatWeights;
	float** firstMoment;
	float** secondMoment;
	float* alphas;
	float* beta1s;
	float* beta2s;
	float* unscaleParameters;
	int* numOverflows;

	T* realGpuGrads;
	T* realGpuWeights;
	float* realFloatGpuWeights;
	float* realGpuFirstMoment;
	float* realGpuSecondMoment;


	const float beta1 = 0.5f;
	const float beta2 = 0.5f;
	const float alpha = 1.0f;
	const float unscaleParameter = 1.0f;

	cudaStream_t s;
	CUDACHECK(cudaStreamCreate(&s));
	size_t* buffSizes;
	size_t* buffIdToParentBufferId;
	size_t* d_parentBuffSizes;

	std::vector<size_t> smallBufferIdToParentBufferId;
	std::vector<size_t> buffSizesAndLayerId;
	size_t totalSize = 0;
	for (size_t i = 0; i < nbuffs; i++) {
		int nSmallBuffersForThisBuffer = h_buffSizes[i] / maxBuffSize;
		if ((h_buffSizes[i] % maxBuffSize) != 0){
			nSmallBuffersForThisBuffer++;
		}
		for (int j = 0; j < nSmallBuffersForThisBuffer; j++){
			buffSizesAndLayerId.push_back((std::min((size_t) (h_buffSizes[i] - j*maxBuffSize), (size_t) maxBuffSize) % (1<<10)) + (j << 10) + ((size_t)nSmallBuffersForThisBuffer << 35));
			smallBufferIdToParentBufferId.push_back(i);
		}
		totalSize += nSmallBuffersForThisBuffer*maxBuffSize;
		// buffSizesAndLayerId[i] = (h_buffSizes[i] % (1<<10)) + (i << 10) + (nbuffs << 20);
	}

	// printf("totalSize padded size is %ld\n", totalSize);

	if (rank == 0) {
		// for (auto i = 0; i < smallBufferIdToParentBufferId.size(); i++) {
		// 	printf("(%ld, %ld)\n", i, smallBufferIdToParentBufferId[i]);
		// }
	}

	float** cpuMomentum = (float**)malloc(nbuffs * sizeof(float*));
	float** cpuVelocity = (float**)malloc(nbuffs * sizeof(float*));
	T** cpuGrads = (T**)malloc(nbuffs * sizeof(T*));
	T** cpuWeight = (T**)malloc(nbuffs * sizeof(T*));
	float** cpuFloatWeight = (float**)malloc(nbuffs * sizeof(float*));

	for (size_t i = 0; i < nbuffs; i++) {
		cpuVelocity[i] = (float*)malloc(h_buffSizes[i]*sizeof(float));
		cpuMomentum[i] = (float*)malloc(h_buffSizes[i]*sizeof(float));
		cpuGrads[i] = (T*)malloc(h_buffSizes[i]*sizeof(T));
		cpuWeight[i] = (T*)malloc(h_buffSizes[i]*sizeof(T));
		cpuFloatWeight[i] = (float*)malloc(h_buffSizes[i]*sizeof(float));
		for (size_t index = 0; index < h_buffSizes[i]; index++){
			cpuVelocity[i][index] = 0.f;
			cpuMomentum[i][index] = 0.f;
			float w = 1.0f;//(float)(i%2 + 1)*(index%20+1);
			cpuGrads[i][index] = __float2half(w);
			cpuWeight[i][index] = __float2half(w);
			cpuFloatWeight[i][index] = w;
		}
	}

	int smallNbuffer = buffSizesAndLayerId.size();
	CUDACHECK(cudaMalloc(&buffSizes, smallNbuffer * sizeof(size_t)));
	CUDACHECK(cudaMemcpy(buffSizes, buffSizesAndLayerId.data(), smallNbuffer * sizeof(size_t), cudaMemcpyHostToDevice));
	CUDACHECK(cudaMalloc(&buffIdToParentBufferId, smallBufferIdToParentBufferId.size() * sizeof(size_t)));
	CUDACHECK(cudaMemcpy(buffIdToParentBufferId, smallBufferIdToParentBufferId.data(), smallBufferIdToParentBufferId.size() * sizeof(size_t), cudaMemcpyHostToDevice));
	
	CUDACHECK(cudaMalloc(&d_parentBuffSizes, sizeof(size_t)*nbuffs));
	CUDACHECK(cudaMemcpy(d_parentBuffSizes, h_buffSizes, nbuffs*sizeof(size_t), cudaMemcpyHostToDevice));
	CUDACHECK(cudaMalloc(&weights, smallNbuffer * sizeof(T*)));
	CUDACHECK(cudaMalloc(&gradients, smallNbuffer * sizeof(T*)));
	CUDACHECK(cudaMalloc(&floatWeights, smallNbuffer * sizeof(float*)));
	
	CUDACHECK(cudaMalloc(&alphas, sizeof(float)));
	CUDACHECK(cudaMalloc(&beta1s, sizeof(float)));
	CUDACHECK(cudaMalloc(&beta2s, sizeof(float)));
	CUDACHECK(cudaMalloc(&unscaleParameters, sizeof(float)));
	CUDACHECK(cudaMalloc(&numOverflows, sizeof(int)));

	CUDACHECK(cudaMemcpy(alphas, &alpha, sizeof(float), cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(beta1s, &beta1, sizeof(float), cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(beta2s, &beta2, sizeof(float), cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(unscaleParameters, &unscaleParameter, sizeof(float), cudaMemcpyHostToDevice));

	T** tmpGrads = (T**)malloc(smallNbuffer * sizeof(T**));
	T** tmpWeights = (T**)malloc(smallNbuffer * sizeof(T**));
	float** tmpFloatWeights = (float**)malloc(smallNbuffer * sizeof(float**));
	float** tmpFirstMoment = (float**)malloc(smallNbuffer * sizeof(float**));
	float** tmpSecondMoment = (float**)malloc(smallNbuffer * sizeof(float**));
	float* singleBufferSecondMoment = nullptr;
	float* singleBufferFirstMoment = nullptr;
	double* weightNorm;
	float* rStorageBuff;

	CUDACHECK(cudaMalloc(&weightNorm, nbuffs * 2 * sizeof(double)));
	CUDACHECK(cudaMemset(weightNorm, 0, nbuffs * 2 * sizeof(double)));


	size_t mSizeForSimple = 0;
	{
		assert (atoi(getenv ("NCCL_MIN_NCHANNELS")) == atoi(getenv ("NCCL_MAX_NCHANNELS")));
		int nChannels = atoi(getenv ("NCCL_MIN_NCHANNELS"));
		int nThreads = atoi(getenv("NCCL_NTHREADS"));
		int channelBuffSize = atoi(getenv("NCCL_BUFFSIZE"));
		const int stepSize = channelBuffSize / (sizeof(T)*NCCL_STEPS);
		const size_t chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
		size_t maxPartSizeForSimple = std::min(chunkSize, DIVUP(totalSize,comm_size*nChannels));
		ALIGN_SIZE(maxPartSizeForSimple, nThreads*sizeof(uint64_t)/sizeof(T));
		const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
		const ssize_t numParts = DIVUP(totalSize, comm_size*loopSize);
		mSizeForSimple = maxPartSizeForSimple*nChannels*numParts;
		CUDACHECK(cudaMalloc(&singleBufferSecondMoment, mSizeForSimple * sizeof(float)));
		CUDACHECK(cudaMalloc(&singleBufferFirstMoment, mSizeForSimple * sizeof(float)));
		CUDACHECK(cudaMemset(singleBufferSecondMoment, 0, mSizeForSimple * sizeof(float)));
		CUDACHECK(cudaMemset(singleBufferFirstMoment, 0, mSizeForSimple * sizeof(float)));
		CUDACHECK(cudaMalloc(&rStorageBuff, mSizeForSimple * sizeof(float)));
	}

	CUDACHECK(cudaMalloc(&realGpuGrads, totalSize * sizeof(T)));
	CUDACHECK(cudaMalloc(&realGpuWeights, totalSize * sizeof(T)));
 	CUDACHECK(cudaMalloc(&realFloatGpuWeights, totalSize * sizeof(float)));

	int accSize = 0;
	int curBuffIndex = 0;
	T** d_tmpParentBuffWeights = new T*[nbuffs];
	T** d_tmpParentBuffGrads = new T*[nbuffs];

	for (size_t i = 0; i < nbuffs; i++){
		int nSmallBuffersForThisBuffer = h_buffSizes[i] / maxBuffSize;
		if ((h_buffSizes[i] % maxBuffSize) != 0){
			nSmallBuffersForThisBuffer++;
		}
		CUDACHECK(cudaMemcpy(&realGpuGrads[accSize], cpuGrads[i], h_buffSizes[i] * sizeof(T), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(&realGpuWeights[accSize], cpuWeight[i], h_buffSizes[i] * sizeof(T), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(&realFloatGpuWeights[accSize], cpuFloatWeight[i], h_buffSizes[i] * sizeof(float), cudaMemcpyHostToDevice));

		d_tmpParentBuffWeights[i] = &realGpuWeights[accSize];
		d_tmpParentBuffGrads[i] = &realGpuGrads[accSize];

		// CUDACHECK(cudaMemcpy(&realGpuFirstMoment[accSize], cpuMomentum[i], h_buffSizes[i] * sizeof(T), cudaMemcpyHostToDevice));
		// CUDACHECK(cudaMemcpy(&realGpuSecondMoment[accSize], cpuVelocity[i], h_buffSizes[i] * sizeof(T), cudaMemcpyHostToDevice));
		for (int j = 0; j < nSmallBuffersForThisBuffer; j++){
			tmpGrads[curBuffIndex] = &realGpuGrads[accSize + j*maxBuffSize];
			tmpWeights[curBuffIndex] = &realGpuWeights[accSize + j*maxBuffSize];
			tmpFloatWeights[curBuffIndex] = &realFloatGpuWeights[accSize + j*maxBuffSize];
			//tmpFirstMoment[curBuffIndex] = &realGpuFirstMoment[accSize + j*maxBuffSize];
			//tmpSecondMoment[curBuffIndex] = &realGpuSecondMoment[accSize + j*maxBuffSize];
			curBuffIndex++;
		}
		
		accSize += nSmallBuffersForThisBuffer * maxBuffSize;
	}

	CUDACHECK(cudaMemcpy(gradients, tmpGrads, sizeof(T*) * smallNbuffer, cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(weights, tmpWeights, sizeof(T*) * smallNbuffer, cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(floatWeights, tmpFloatWeights, sizeof(float*) * smallNbuffer, cudaMemcpyHostToDevice));

	float** tmpOldWeights = new float*[nbuffs];
	float** tmpNewWeights = new float*[nbuffs];
	float** tmpOldGrads = new float*[nbuffs];
    float** tmpOldFirstMomentum = new float*[nbuffs];
	float** tmpOldSecondMomentum = new float*[nbuffs];
	float** tmpOldFloatWeights = new float*[nbuffs];

	for (size_t buff = 0; buff < nbuffs; buff++) {
		tmpOldWeights[buff] = new float[h_buffSizes[buff]];
		tmpNewWeights[buff] = new float[h_buffSizes[buff]];
		tmpOldGrads[buff] = new float[h_buffSizes[buff]];
		tmpOldFirstMomentum[buff] = new float[h_buffSizes[buff]];
		tmpOldSecondMomentum[buff] = new float[h_buffSizes[buff]];
		memset(tmpOldFirstMomentum[buff], 0, h_buffSizes[buff]*sizeof(float));
		memset(tmpOldSecondMomentum[buff], 0, h_buffSizes[buff]*sizeof(float));
	}

	
	MPI_Barrier(MPI_COMM_WORLD);
	
	float totalTime = 0.0;

	for (int iter = 0; iter < epochs + 5; iter++) {
    	// printf("iter %d\n", iter);
		if (check_results and iter < 1) {
			for (size_t buff = 0; buff < nbuffs; buff++) {
				int thisBufferSize = h_buffSizes[buff];
        		cudaMemcpyHalfDevice2FloatHost(tmpOldWeights[buff], d_tmpParentBuffWeights[buff], thisBufferSize);
        		cudaMemcpyHalfDevice2FloatHost(tmpOldGrads[buff], d_tmpParentBuffGrads[buff], thisBufferSize);
        		//cudaMemcpy(tmpOldFirstMomentum[buff], tmpFirstMoment[buff], sizeof(T)*thisBufferSize, cudaMemcpyDeviceToHost);
        		//cudaMemcpy(tmpOldSecondMomentum[buff], tmpSecondMoment[buff], sizeof(T)*thisBufferSize, cudaMemcpyDeviceToHost);
			}
		}

		cudaEvent_t start, stop;
		CUDACHECK(cudaEventCreate(&start));
		CUDACHECK(cudaEventCreate(&stop));
		CUDACHECK(cudaEventRecord(start,0));
		NCCLCHECK(ncclAllReduceScatteredLAMB((const void**)gradients, (void**)weights, floatWeights, (void**)singleBufferFirstMoment, 
											 (void**)singleBufferSecondMoment, smallNbuffer, buffSizes,
											 totalSize,d_parentBuffSizes, (void*) alphas, (void*) beta1s, (void*)beta2s,(void*)unscaleParameters, numOverflows,
											 weightNorm, rStorageBuff, nbuffs, buffIdToParentBufferId,
											 iter, datatype, ncclSum, comm, s));
		CUDACHECK(cudaEventRecord(stop,0));
		CUDACHECK(cudaEventSynchronize(stop));
		float elapsedTime;
		CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
		if (iter >= 5) 
			totalTime += elapsedTime;

		if (check_results and iter < 1) {
			printf("checking results\n");
			bool passed = true;		
			for (size_t buff = 0; buff < nbuffs; buff++) {
				int thisBufferSize = h_buffSizes[buff];
				cudaMemcpyHalfDevice2FloatHost(tmpNewWeights[buff], d_tmpParentBuffWeights[buff], thisBufferSize);
				passed = check_epoch_results_lamb<float>(rank, 2, buff, iter, h_buffSizes[buff], tmpOldGrads[buff], tmpOldWeights[buff], tmpNewWeights[buff], tmpOldFirstMomentum[buff], tmpOldSecondMomentum[buff], beta1, beta2, alpha, (T)1e-6);
				if (!passed)
					break;
			}
			if (passed)
				printf("Results checks out!\n");
			else {
				printf("Correctness failed!\n");
				abort();
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	if (rank == 0) {
		printf("{SZ: %ld, Epochs: %d, Time: %f}\n", totalSize, epochs, totalTime);
	}
	return totalTime/epochs;
}

int main(int argc, char* argv[])
{
	  MPI_Init(&argc, &argv);

	
	if (argc < 2) {
		printf("provide optimizer type: adam or lamb\n");
		return 0;
	}

	// if (argc < 3) {
	// 	printf ("Provide four arguments: size scattered/single\n");
	// 	return 0;
	// }

	size_t size = 0;  
	const char* opt_type = argv[1];

	float elapsedTime;
	if(true) {
		// size_t nbuffs = size/maxBuffSize;
		// if ((size % maxBuffSize) != 0)
		// 	nbuffs++;
		// size_t* buffSizes = new size_t[nbuffs];
		// size_t curSize = 0;
		// for (size_t i = 0; i < nbuffs; i++) {
		// 	buffSizes[i] = (size_t) std::min((size_t)maxBuffSize, size-curSize);
		// 	curSize += buffSizes[i];
		// }
		// const size_t nbuffs = 396;
	  // const size_t buffSizes[] = { 31260672, 2048, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 30528, 1024, 1024, 1024 };
		const size_t buffSizes[] = {78151680, 1310720, 5120, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 
																10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 6553600, 2560, 2560, 2560, 10485760, 4096, 10485760, 2560, 2560, 2560, 6553600, 2560, 30528, 6553600, 2560, 2560, 2560, 5120, 2};
		//const size_t buffSizes[] = { 31260672, 2048, 1048576};
    const size_t nbuffs = sizeof(buffSizes)/sizeof(buffSizes[0]);
		size = 0;
		for (size_t i = 0; i < nbuffs;i++) {
			size += buffSizes[i];
		}
		if (strcmp(opt_type, "adam") == 0) {
			elapsedTime = run_scattered_adam_mp(nbuffs, buffSizes, ncclHalf, false);
		} else {
			elapsedTime = run_scattered_lamb(nbuffs, buffSizes, ncclHalf, true);
		}
		//elapsedTime = run_scattered_adam_mp(nbuffs, buffSizes, ncclHalf, true);
		// elapsedTime = run_scattered_lamb(nbuffs, buffSizes, ncclHalf, false);
		// delete[] buffSizes;
	} else {
		// elapsedTime = run_adam<float>(algo, (bool)check_results, size, (float)gradient_factor, ncclFloat, weight_update_time, reduce_time);
	}
	// int comm_size;
  	// MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	// printf ("Elapsed Time: %f BW %f\n", elapsedTime, (size*sizeof(float)*2.*(comm_size-1))/comm_size/(elapsedTime/1000.)/1e9);
MPI_Finalize();
	return 0;
}
