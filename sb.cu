
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define WARP  8

__global__ void incrementArrayOnDevice(int *a, int*b, int*r, int N)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int res = 0;

  int warp = idx / WARP;

  // the two threads that are going to interract are idx and idx + WARP
  if (warp%2 ==0)
    {
      a[idx] = 1;
      res = b[idx];
      __threadfence();
      r[idx] = res;
    }
  else
    {
      b[idx - WARP] = 1;
      res = a[idx - WARP];
      __threadfence();
      r[idx] = res;
    }
}

int main(void)
{
  int *a_h, *b_h, *r_h;     // pointers to host memory
  int *a_d, *b_d, *r_d;     // pointers to device memory

  int N = WARP * 10 ;
  int i;
  size_t size = N*sizeof(int);

  // allocate arrays on host
  a_h = (int *)malloc(size);
  b_h = (int *)malloc(size);
  r_h = (int *)malloc(size);

  // allocate arrays on device
  cudaMalloc((void **) &a_d, size);
  cudaMalloc((void **) &b_d, size);
  cudaMalloc((void **) &r_d, size);


  int finished = 0;
  int iteration = 0;
  int witnesses = 0;

  while (!finished)
    {
      // initialize host data
      for (i=0; i<N; i++) {
	a_h[i] = 0;
	b_h[i] = 0;
	r_h[i] = 0;
      }
      
      // send data from host to device: a_h to a_d 
      cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
      cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
      cudaMemcpy(r_d, r_h, size, cudaMemcpyHostToDevice);
      
      // do calculation on device:
      // Part 1 of 2. Compute execution configuration
      int blockSize = 4;
      int nBlocks = N/blockSize + (N%blockSize == 0?0:1);
      // Part 2 of 2. Call incrementArrayOnDevice kernel 
      incrementArrayOnDevice <<< nBlocks, blockSize >>> (a_d, b_d, r_d, N);
      
      cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
      cudaMemcpy(b_h, b_d, size, cudaMemcpyDeviceToHost);
      cudaMemcpy(r_h, r_d, size, cudaMemcpyDeviceToHost);
      
      // check result
      for (i=0; i< (N - WARP); i++)
	{
	  if (i < N- WARP && r_h[i] == 0 && r_h[i+WARP] == 0)
	    {
	      finished = 1;
	      break;
	    }
	};
      iteration ++;
    }
  // count the number of consecutive witnesses
  for (int j = i; j < N - WARP; j ++)
    {if (r_h[j] == 0 && r_h[j+WARP] == 0)
	witnesses ++;
    }


  printf("found witness after %i iterations\n",iteration);
  printf("%i witnesses (first: %i), N= %i\n",witnesses,i,N);

  // cleanup
  free(a_h); free(b_h); free(r_h);
  cudaFree(a_d); cudaFree(b_d); cudaFree(r_d);
}
