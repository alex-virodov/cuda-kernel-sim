#include    "wb.h"
#include <stdio.h>

#define BLOCK_SIZE 4 //@@ You can change this

// Test code that adds +1.0f to an input 2d array 
void code2(float* input, float* output, int len) 
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x);
	int j = (threadIdx.y + blockIdx.y * blockDim.y);
	int idx = i + j * len;
	if (idx < len*len) { output[idx] = input[idx]+1; }
	printf("== tid.x=%d, tid.y=%d, bid.x=%d, bid.y=%d, idx=%d\n", 
		threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, idx);
}

// rename to main to enable
int zzmain() 
{
	const int size = 11;
	float* in  = new float[size*size];
	float* out = new float[size*size];

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			in[i*size + j] = (float)(j+1 + 100*i); 
		};
	}
	
	// Make y-dimension different than x for testing
    dim3 dimGrid (size/BLOCK_SIZE+1, size/(BLOCK_SIZE/2)+1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE/2, 1);

	// Cannot support CUDA's <<<x,y>>> syntax.
	schedule(code2, in, out, size)
		.setBlockSize(dimBlock)
		.setGridSize(dimGrid)
		.run();

	for (int i = 0; i < size*size; i++) {
		printf("%0.2f %0.2f\n", in[i], out[i]);
	}



}