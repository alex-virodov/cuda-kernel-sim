#include    "wb.h"
#include <stdio.h>

#define BLOCK_SIZE 4 //@@ You can change this

// Test code that adds +1.0f to an input 3d array 
void code2(float* input, float* output, int len) 
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x);
	int j = (threadIdx.y + blockIdx.y * blockDim.y);
	int k = (threadIdx.z + blockIdx.z * blockDim.z);
	int idx = i + j * len + k * len * len;
	if (i < len && j < len && k < len) { 
		output[idx] = input[idx]+1; 
	} else {
		idx = -1;
	}
	printf("== tid.x=%d, tid.y=%d, tid.z=%d, bid.x=%d, bid.y=%d, bid.z=%d, idx=%d\n", 
		threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, idx);
}

// rename to main to enable
int zzmain() 
{
	const int size = 3;
	float* in  = new float[size*size*size];
	float* out = new float[size*size*size];

	for (int k = 0; k < size; k++) {
		for (int j = 0; j < size; j++) {
			for (int i = 0; i < size; i++) {
				in[i + j*size + k*size*size] = (float)(i+1 + 10*j + 100*k); 
			};
		}
	}
	
	// Make y-dimension different than x for testing
    dim3 dimGrid (size/BLOCK_SIZE+1, size/(BLOCK_SIZE/2)+1, size/(BLOCK_SIZE/2)+1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE/2, BLOCK_SIZE/2);

	// Cannot support CUDA's <<<x,y>>> syntax.
	schedule(code2, in, out, size)
		.setBlockSize(dimBlock)
		.setGridSize(dimGrid)
		.run();

	for (int i = 0; i < size*size*size; i++) {
		printf("%0.2f %0.2f\n", in[i], out[i]);
	}



}