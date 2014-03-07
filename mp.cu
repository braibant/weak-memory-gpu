
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define WARP  32

__global__ void incrementArrayOnDevice(int *a, int*b, int*ra, int*rb, int N)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int resa = -2;
  int resb = -2;

  int warp = idx / WARP;

  // the two threads that are going to interract are idx and idx + WARP
  if (warp%2 ==0)
    {
      a[idx] = 1;
      b[idx] = 1;
    }
  else
    {
      resb = b[idx - WARP];
      resa = a[idx - WARP];
      // __threadfence(); 
      ra[idx] = resa;
      rb[idx] = resb;
    }
}

int main(void)
{
  int *a_h, *b_h, *ra_h, *rb_h;     // pointers to host memory
  int *a_d, *b_d, *ra_d, *rb_d;     // pointers to device memory

  int N = WARP * 100 ;
  int i;
  size_t size = N*sizeof(int);

  // allocate arrays on host
  a_h = (int *)malloc(size);
  b_h = (int *)malloc(size);
  ra_h = (int *)malloc(size);
  rb_h = (int *)malloc(size);

  // allocate arrays on device
  cudaMalloc((void **) &a_d, size);
  cudaMalloc((void **) &b_d, size);
  cudaMalloc((void **) &ra_d, size);
  cudaMalloc((void **) &rb_d, size);


  int finished = 0;
  int iteration = 0;

  while (!finished)
    {
      // initialize host data
      for (i=0; i<N; i++) {
	a_h[i] = 0;
	b_h[i] = 0;
	ra_h[i] = -1;
	rb_h[i] = -1;
      }
      
      // send data from host to device: a_h to a_d 
      cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
      cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
      cudaMemcpy(ra_d, ra_h, size, cudaMemcpyHostToDevice);
      cudaMemcpy(rb_d, rb_h, size, cudaMemcpyHostToDevice);
      
      // do calculation on device:
      // Part 1 of 2. Compute execution configuration
      int blockSize = WARP;
      int nBlocks = N/blockSize + (N%blockSize == 0?0:1);
      // Part 2 of 2. Call incrementArrayOnDevice kernel 
      incrementArrayOnDevice <<< nBlocks, blockSize >>> (a_d, b_d, ra_d, rb_d, N);
      
      cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
      cudaMemcpy(b_h, b_d, size, cudaMemcpyDeviceToHost);
      cudaMemcpy(ra_h, ra_d, size, cudaMemcpyDeviceToHost);
      cudaMemcpy(rb_h, rb_d, size, cudaMemcpyDeviceToHost);
      
      /* for (i=WARP; i< N; i++) */
      /* 	{ */
      /* 	  fprintf(stdout,"iter=%03i,\tra=%i, rb=%i\n", i, ra_h[i], rb_h[i]); */
      /* 	}; */
      
      // check result
      for (i=WARP; i< N; i++)
	{
	  if (rb_h[i] == 1 && ra_h[i] == 0) {finished = 1; break;} 	
	};
      iteration ++;
    }

  printf("found witness after %i iterations\n",iteration);

  // cleanup
  free(a_h); free(b_h); free(ra_h); free(rb_h);
  cudaFree(a_d); cudaFree(b_d); cudaFree(ra_d); cudaFree(rb_d);
}
