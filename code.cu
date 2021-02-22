// Compile with:
// nvcc -O3 code.cu -o run.out -lcublas

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 16;

// Array size must be divisible by BLOCK_SIZE:
const int xdim      = 160;
const int ydim      = 320;

const int maxItr   = 50000;
const int stepskip = 500;	
const float tol    = 0.001;

/* --------------------------------------------------------------------------------------------------------------- */

__global__ void udiff(float* udiff_d, float* uold_d, float* unew_d, int SizeX, int SizeY)
{
  int tx=threadIdx.x+1;
  int ty=threadIdx.y+1;
  int bx=blockIdx.x * blockDim.x;
  int by=blockIdx.y * blockDim.y;
  int x=tx+bx;  
  int y=ty+by;
  __shared__ float u_sh1[BLOCK_SIZE_X+2][BLOCK_SIZE_Y+2];
  u_sh1[tx][ty]=uold_d[x+y*SizeX];
  __shared__ float u_sh2[BLOCK_SIZE_X+2][BLOCK_SIZE_Y+2];
  u_sh2[tx][ty]=unew_d[x+y*SizeX];
  
  if (tx == 1)
  { 
	  u_sh1[0][ty]=uold_d[x-1+y*SizeX];
	  u_sh2[0][ty]=unew_d[x-1+y*SizeX];
  }
  
  if (tx == BLOCK_SIZE_X)
  { 
	  u_sh1[BLOCK_SIZE_X+1][ty]=uold_d[x+1+y*SizeX];
          u_sh2[BLOCK_SIZE_X+1][ty]=unew_d[x+1+y*SizeX];
  }

  if (ty == 1)
  {
	  u_sh1[tx][0]=uold_d[x+(y-1)*SizeX];
	  u_sh2[tx][0]=unew_d[x+(y-1)*SizeX];
  }
  
  if (ty == BLOCK_SIZE_Y)
  { 
	  u_sh1[tx][BLOCK_SIZE_Y+1]=uold_d[x+(y+1)*SizeX];
	  u_sh2[tx][BLOCK_SIZE_Y+1]=unew_d[x+(y+1)*SizeX];
  }

  __syncthreads();
  udiff_d[(x-1)+(y-1)*(SizeX-2)] = u_sh2[tx][ty]-u_sh1[tx][ty];
}

/* --------------------------------------------------------------------------------------------------------------- */

__global__ void Jacobi(float* uold_d, float* unew_d, int SizeX, int SizeY, float h)
{
  int tx=threadIdx.x+1; 
  int ty=threadIdx.y+1;
  int bx=blockIdx.x*blockDim.x;
  int by=blockIdx.y*blockDim.y;
  int x=tx+bx;  
  int y=ty+by;
  __shared__ float u_sh[BLOCK_SIZE_X+2][BLOCK_SIZE_Y+2];
  u_sh[tx][ty]=uold_d[x+y*SizeX];
  if (tx == 1) u_sh[0][ty]=uold_d[x-1+y*SizeX];
  if (tx == BLOCK_SIZE_X) u_sh[BLOCK_SIZE_X+1][ty]=uold_d[x+1+y*SizeX];
  if (ty == 1) u_sh[tx][0]=uold_d[x+(y-1)*SizeX];
  if (ty == BLOCK_SIZE_Y) u_sh[tx][BLOCK_SIZE_Y+1]=uold_d[x+(y+1)*SizeX];
  __syncthreads();
  unew_d[x+y*SizeX] = 0.25f*(u_sh[tx+1][ty]+u_sh[tx-1][ty]+u_sh[tx][ty+1]+u_sh[tx][ty-1]);
}

/* --------------------------------------------------------------------------------------------------------------- */

void initialize(float *u, int xsize, int ysize)
{
  for(int i=0; i < ysize; i++)
  {
    u[(ysize-i-1)*xsize] = (100.0/(ysize-1.0))*(i);  // upper boundary (0 to 100, from right to left)
    u[i*xsize+xsize-1] = (100.0/(ysize-1.0))*(i);    // lower boundary (0 to 100, from left to right)
  }
  for(int j=0; j< xsize; j++)
  {
    u[xsize-j-1] = (100.0/(xsize-1.0))*(j);         // left boundary (0 to 100, from bottom to top)
    u[(ysize-1)*xsize+j] = (100.0/(xsize-1.0))*(j); // right boundary (0 to 100, from top to bottom)
  }

  // some simpler boundaries (all constant)
  // for(int i=0; i < ysize; i++)
  // {
  //   u[i*xsize] = 1.0; // upper boundary
  //   u[i*xsize+xsize-1] = 0.0; // lower boundary
  // }
  // for(int j=0; j< xsize; j++)
  // {
  //   u[j] = 1.0; // left boundary
  //   u[(ysize-1)*xsize+j] = 0.0; // right boundary
  // }

}

/* --------------------------------------------------------------------------------------------------------------- */

void output(float *u, int xsize, int ysize, int step)
{
  char filename[30]; 
  sprintf(filename, "solution_iter_%d.txt", step);  
  FILE *fp = fopen(filename,"wt");
  for (int i=0; i<xsize; i++)
  {
    for (int j=0; j<ysize; j++)
    {
      fprintf(fp," %f",u[j*xsize+i]);
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
}

/* --------------------------------------------------------------------------------------------------------------- */

int main(void)
{
  // double time_spent = 0.0;
  // clock_t begin = clock();
  float maxerr = 100.0;
  float *u_h; 
  int ArraySizeX = xdim+2;
  int ArraySizeY = ydim+2;
  size_t size = ArraySizeX*ArraySizeY*sizeof(float); // size of flattened array
  u_h = (float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));
  initialize(u_h, ArraySizeX, ArraySizeY);
  float *unew_d, *uold_d, *udiff_d;
  cudaMalloc(&uold_d,size);
  cudaMalloc(&unew_d,size);
  cudaMalloc(&udiff_d,(ArraySizeX-2)*(ArraySizeY-2)*sizeof(float));
  cudaMemcpy(uold_d, u_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(unew_d, u_h, size, cudaMemcpyHostToDevice);
  cublasHandle_t handle;
  cublasCreate(&handle);
  int nBlocksX=(ArraySizeX-2)/BLOCK_SIZE_X;
  int nBlocksY=(ArraySizeY-2)/BLOCK_SIZE_Y;
  dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
  dim3 dimGrid(nBlocksX,nBlocksY);

  int nsteps = 0;
  output(u_h, ArraySizeX, ArraySizeY, nsteps);
  while (nsteps < maxItr)
  {
    nsteps++;
    Jacobi<<<dimGrid, dimBlock>>>(uold_d, unew_d, ArraySizeX, ArraySizeY, 1.0);
    udiff<<<dimGrid, dimBlock>>>(udiff_d, uold_d, unew_d, ArraySizeX, ArraySizeY);
    int index_max;
    cublasIsamax(handle, (ArraySizeX-2)*(ArraySizeY-2), udiff_d, 1, &index_max);
    maxerr = 0.0;
    cudaMemcpy(&maxerr, &(udiff_d[index_max-1]), sizeof(float), cudaMemcpyDeviceToHost); 
  
    if ( !(nsteps % stepskip) )
    {
      printf("At iteration %d, the max error belongs to element %d with magnitude of %g\n", nsteps, index_max, fabs(maxerr));
      cudaMemcpy(u_h, uold_d, size, cudaMemcpyDeviceToHost);
      output(u_h, ArraySizeX, ArraySizeY, nsteps);
    }
  
    if (maxerr < tol)
    {
      cudaMemcpy(u_h, uold_d, size, cudaMemcpyDeviceToHost);
      printf("Converged after %d iterations.\n", nsteps);
      output(u_h, ArraySizeX, ArraySizeY, nsteps);
      cublasDestroy(handle);
      cudaFree(uold_d);
      cudaFree(unew_d);
      free(u_h);
      // clock_t end = clock();
      // time_spent += (double)(end - begin)/ CLOCKS_PER_SEC;
      break;
    }
  
    float *tmp = uold_d;
    uold_d = unew_d;
    unew_d = tmp;
  }

  if (maxerr > tol)
  {
    printf("Not converged after %d iterations.\n", nsteps);
    cublasDestroy(handle);
    cudaFree(uold_d);
    cudaFree(unew_d);
    free(u_h);
    // clock_t end = clock();
    // time_spent += (double)(end - begin)/ CLOCKS_PER_SEC;
  }
}