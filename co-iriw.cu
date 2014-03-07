
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define WARP  8
#define WARP2 16
#define WARP3 24

__global__ void incrementArrayOnDevice(int *x, int*y, int*ra, int*rb, int N)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int r0 = -2;
  int r1 = -2;

  int warp = idx / WARP;
 
  int div = warp % 4;
  if (div == 0) {x[idx] = 1;}
  else if (div == 2) {x[idx - WARP2] = 2;}
  else if (div == 1) {
    y [idx] = 1;
    r0 = x[idx - WARP];
    ra[idx-WARP] = r0;
    r1 = x[idx - WARP];
    rb[idx-WARP] = r1;}
  else
    {
      r0 = x[idx - WARP3];
      ra[idx-WARP2] = r0; // notice the warp2
      r1 = x[idx - WARP3];
      rb[idx-WARP2] = r1;}
  /*
  switch (warp % 4)
   {
   case 0:
     y[idx + WARP] = 1;
     x[idx] = 1; break;
   case 2:
     x[idx - WARP2] = 2; y [idx] = 1; break;
   case 1:
     r0 = x[idx - WARP];
     r1 = x[idx - WARP];
     ra[idx-WARP] = r0;
     rb[idx-WARP] = r1;
     break;
   case 3:
     r0 = x[idx - WARP3];
     r1 = x[idx - WARP3];
     ra[idx-WARP2] = r0; // notice the warp2
     rb[idx-WARP2] = r1;
     break;
     };*/
}

inline static int final_cond(int _out_1_r0,int _out_1_r1,int _out_3_r0,int _out_3_r1) {
int cond;
cond = (((_out_1_r0 == 2) && (((_out_1_r1 == 2) && ((((_out_3_r0 == 2) && ((_out_3_r1 == 2) || (_out_3_r1 == 1))) || ((_out_3_r0 == 1) && ((_out_3_r1 == 2) || (_out_3_r1 == 1)))) || ((_out_3_r0 == 0) && (((_out_3_r1 == 2) || (_out_3_r1 == 1)) || (_out_3_r1 == 0))))) || ((_out_1_r1 == 1) && ((((_out_3_r0 == 2) && ((_out_3_r1 == 2) || (_out_3_r1 == 1))) || ((_out_3_r0 == 1) && (_out_3_r1 == 1))) || ((_out_3_r0 == 0) && (((_out_3_r1 == 2) || (_out_3_r1 == 1)) || (_out_3_r1 == 0))))))) || ((_out_1_r0 == 1) && (((_out_1_r1 == 2) && ((((_out_3_r0 == 2) && (_out_3_r1 == 2)) || ((_out_3_r0 == 1) && ((_out_3_r1 == 2) || (_out_3_r1 == 1)))) || ((_out_3_r0 == 0) && (((_out_3_r1 == 2) || (_out_3_r1 == 1)) || (_out_3_r1 == 0))))) || ((_out_1_r1 == 1) && ((((_out_3_r0 == 2) && ((_out_3_r1 == 2) || (_out_3_r1 == 1))) || ((_out_3_r0 == 1) && ((_out_3_r1 == 2) || (_out_3_r1 == 1)))) || ((_out_3_r0 == 0) && (((_out_3_r1 == 2) || (_out_3_r1 == 1)) || (_out_3_r1 == 0)))))))) || ((_out_1_r0 == 0) && ((((_out_1_r1 == 2) && ((((_out_3_r0 == 2) && ((_out_3_r1 == 2) || (_out_3_r1 == 1))) || ((_out_3_r0 == 1) && ((_out_3_r1 == 2) || (_out_3_r1 == 1)))) || ((_out_3_r0 == 0) && (((_out_3_r1 == 2) || (_out_3_r1 == 1)) || (_out_3_r1 == 0))))) || ((_out_1_r1 == 1) && ((((_out_3_r0 == 2) && ((_out_3_r1 == 2) || (_out_3_r1 == 1))) || ((_out_3_r0 == 1) && ((_out_3_r1 == 2) || (_out_3_r1 == 1)))) || ((_out_3_r0 == 0) && (((_out_3_r1 == 2) || (_out_3_r1 == 1)) || (_out_3_r1 == 0)))))) || ((_out_1_r1 == 0) && ((((_out_3_r0 == 2) && ((_out_3_r1 == 2) || (_out_3_r1 == 1))) || ((_out_3_r0 == 1) && ((_out_3_r1 == 2) || (_out_3_r1 == 1)))) || ((_out_3_r0 == 0) && (((_out_3_r1 == 2) || (_out_3_r1 == 1)) || (_out_3_r1 == 0)))))));
return cond;
}

int main(void)
{
  int *a_h, *b_h, *ra_h, *rb_h;     // pointers to host memory
  int *a_d, *b_d, *ra_d, *rb_d;     // pointers to device memory

  int N = WARP * 1000 ;
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
      if (iteration % 1000 ==0) {printf("iteration:%i\n", iteration);};
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
      
      // check result
      for (i=0; i< N/(WARP * 4); i++)
	{ 
	  for (int j = 0; j < WARP; j ++)
	    {
	      int k = i * WARP * 4 + j;
	      if ((ra_h[k] != rb_h[k]) || ra_h[k+WARP] != rb_h[k+ WARP])
		printf ("%i: %i %i %i %i \n", k, ra_h[k],rb_h[k], ra_h[k + WARP],rb_h[k+WARP]);
	      if (!(final_cond(ra_h[k],rb_h[k], ra_h[k + WARP],rb_h[k+WARP]))) {finished = 1;}
	    }
	};
      iteration ++;
    }

  printf("found witness after %i iterations\n",iteration);

  // cleanup
  free(a_h); free(b_h); free(ra_h); free(rb_h);
  cudaFree(a_d); cudaFree(b_d); cudaFree(ra_d); cudaFree(rb_d);
}

