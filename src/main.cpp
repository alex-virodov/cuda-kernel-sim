#include    "wb.h"
#include <stdio.h>

#define BLOCK_SIZE 8 //@@ You can change this

// Test code that adds +1.0f to an input array using a convoluted way to
// test capabilities.
void code(float* input, float* output, int len) 
{
	__shared__ float data[BLOCK_SIZE*2];

	for (int offset = 0; offset < len; offset += blockDim.x*2) 
	{
		printf("==tid %d, offset=%d\n", threadIdx.x, offset);
		int i   = threadIdx.x*2;
		int idx = i + offset;
	
		// Load data
		float u = (idx < len ? input[idx]   : 0);
		float v = (idx < len ? input[idx+1] : 0);

		data[i]   = u;
		data[i+1] = u + v;

		data[i]   += 1.0f;
		data[i+1] += 1.0f;

		// store output
		__syncthreads();
		printf("==tid %d, offset=%d\n", threadIdx.x, offset);
		if (i+offset   < len) { output[i+offset]   = data[i];  }
		if (i+1+offset < len) { output[i+1+offset] = data[i+1];}
	}
}

int main() 
{
	const int size = 33;
	float* in  = new float[size];
	float* out = new float[size];

	for (int i = 0; i < size; i++) { in[i] = (float)(i+1); };
	
    dim3 dimGrid (1, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

	// Cannot support CUDA's <<<x,y>>> syntax.
	schedule(code, in, out, size)
		.setBlockSize(dimBlock)
		.setGridSize(dimGrid)
		.run();

	for (int i = 0; i < size; i++) {
		printf("%0.2f %0.2f\n", in[i], out[i]);
	}



}