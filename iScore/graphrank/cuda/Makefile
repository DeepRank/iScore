CUDA_ROOT = /usr/local/cuda-9.0
CUDA_INC = -I${CUDA_ROOT}/include/
CUDA_LIB = -L${CUDA_ROOT}/lib64/ 


test : main.o cuda_kernel.o
	nvcc ${CUDA_INC} main.o cuda_kernel.o -o test ${CUDA_LIB} -lcuda -lcudart

cuda_kernel.o: cuda_kernel.cu
	nvcc -c $< -o $@ 

main.o : main.c
	nvcc -c  ${CUDA_INC} $< -o $@ ${CUDA_LIB} -lcuda -lcudart

clean :
	rm test cuda_kernel.o main.o 
