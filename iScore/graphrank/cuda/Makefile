CUDA_ROOT = /usr/local/cuda-9.0
CUDA_INC = -I${CUDA_ROOT}/include/
CUDA_LIB = -L${CUDA_ROOT}/lib64/ 
CXX_FLAGS = -std=c++11

test : main.o cuda_kernel.o
	nvcc ${CXX_FLAGS} ${CUDA_INC} main.o cuda_kernel.o -o test ${CUDA_LIB} -lcuda -lcudart

cuda_kernel.o: cuda_kernel.cu
	nvcc ${CXX_FLAGS} -c $< -o $@ 

main.o : main.cu
	nvcc ${CXX_FLAGS} -c  ${CUDA_INC} $< -o $@ ${CUDA_LIB} -lcuda -lcudart

clean :
	rm test cuda_kernel.o main.o 
