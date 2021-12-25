all: single-process-wX multi-process-wX single-process-wX single-process-adam-scattered multi-process-adam-scattered

debug: single-process-wX-debug

single-process-adam-scattered: single-process-adam-scattered.cu
	nvcc $< -I../build/include/ -L../build/lib/ -lnccl -lcublas -o $@ -Xcompiler -Wall -arch=compute_70 -code=sm_70 -lcurand -G -g -std=c++11

single-process-wX: single-process-wX.cu
	nvcc $< -I../build/include/ -L../build/lib/ -lnccl -lcublas -o $@ -Xcompiler -Wall -arch=compute_70 -code=sm_70 -lcurand -G -g -std=c++11

multi-process-wX: multi-process-wX.cpp
	mpicxx $< -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -o $@ -Wall -lcurand -std=c++11

multi-process-adam-scatteredf16: multi-process-adam-scattered.cpp
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I.. -I/usr/local/cuda/include/ -I ../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ \
	-lnccl -lcublas -lcurand -c && mpicxx multi-process-adam-scattered.o -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall \
	-lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

single-process-wX-debug: single-process-wX.cu
	nvcc $< -I../build/include/ -L../build/lib/ -lnccl -lcublas -o $@ -G -g -lcurand -std=c++11

clean:
	rm single-process-wX multi-process-wX single-process-adam-scattered
