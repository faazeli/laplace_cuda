// Jacobi solver for Laplace's equation on a 2-D grid, on the GPU (CUDA + cuBLAS),
// with a CPU reference implementation for verification and speedup measurement.
//
// Build:   nvcc -O3 laplace.cu -o laplace -lcublas
// Run:     ./laplace [--nx 160] [--ny 320] [--tol 1e-3] [--max-iter 200000]
//                    [--check-every 100] [--dump-every 0] [--cpu]
//
// Grid sizes (interior points) must be multiples of the 16x16 thread block.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 16;

#define CUDA_CHECK(call)                                                                       \
  do {                                                                                         \
    cudaError_t err_ = (call);                                                                 \
    if (err_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
      exit(1);                                                                                 \
    }                                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                       \
  do {                                                                           \
    cublasStatus_t st_ = (call);                                                 \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

struct Options {
  int xdim = 160, ydim = 320;  // interior points
  float tol = 1e-3f;
  int maxItr = 200000;
  int checkEvery = 100;  // convergence check interval (each check syncs with the host)
  int dumpEvery = 0;     // write solution_iter_*.txt every N iterations (0 = only the final field)
  bool cpu = false;      // also run the CPU reference and compare
};

/* --------------------------------------------------------------------------------------------------------------- */
// Kernels. Each 16x16 block loads its tile plus a one-cell halo into shared memory.

__global__ void udiff(float* udiff_d, float* uold_d, float* unew_d, int SizeX, int SizeY)
{
  int tx = threadIdx.x + 1;
  int ty = threadIdx.y + 1;
  int x = tx + blockIdx.x * blockDim.x;
  int y = ty + blockIdx.y * blockDim.y;
  // Only the centre cells are needed for the difference, so no halo is loaded here.
  udiff_d[(x - 1) + (y - 1) * (SizeX - 2)] = unew_d[x + y * SizeX] - uold_d[x + y * SizeX];
}

__global__ void Jacobi(float* uold_d, float* unew_d, int SizeX, int SizeY)
{
  int tx = threadIdx.x + 1;
  int ty = threadIdx.y + 1;
  int x = tx + blockIdx.x * blockDim.x;
  int y = ty + blockIdx.y * blockDim.y;
  __shared__ float u_sh[BLOCK_SIZE_X + 2][BLOCK_SIZE_Y + 2];
  u_sh[tx][ty] = uold_d[x + y * SizeX];
  if (tx == 1) u_sh[0][ty] = uold_d[x - 1 + y * SizeX];
  if (tx == BLOCK_SIZE_X) u_sh[BLOCK_SIZE_X + 1][ty] = uold_d[x + 1 + y * SizeX];
  if (ty == 1) u_sh[tx][0] = uold_d[x + (y - 1) * SizeX];
  if (ty == BLOCK_SIZE_Y) u_sh[tx][BLOCK_SIZE_Y + 1] = uold_d[x + (y + 1) * SizeX];
  __syncthreads();
  unew_d[x + y * SizeX] = 0.25f * (u_sh[tx + 1][ty] + u_sh[tx - 1][ty] + u_sh[tx][ty + 1] + u_sh[tx][ty - 1]);
}

/* --------------------------------------------------------------------------------------------------------------- */

void initialize(float* u, int xsize, int ysize)
{
  // Linearly varying Dirichlet boundaries (0 to 100) on all four sides.
  for (int i = 0; i < ysize; i++) {
    u[(ysize - i - 1) * xsize] = (100.0f / (ysize - 1.0f)) * i;
    u[i * xsize + xsize - 1] = (100.0f / (ysize - 1.0f)) * i;
  }
  for (int j = 0; j < xsize; j++) {
    u[xsize - j - 1] = (100.0f / (xsize - 1.0f)) * j;
    u[(ysize - 1) * xsize + j] = (100.0f / (xsize - 1.0f)) * j;
  }
}

void output(const float* u, int xsize, int ysize, int step)
{
  char filename[64];
  snprintf(filename, sizeof filename, "solution_iter_%d.txt", step);
  FILE* fp = fopen(filename, "wt");
  if (!fp) { perror(filename); return; }
  for (int i = 0; i < xsize; i++) {
    for (int j = 0; j < ysize; j++) fprintf(fp, " %f", u[j * xsize + i]);
    fprintf(fp, "\n");
  }
  fclose(fp);
}

/* --------------------------------------------------------------------------------------------------------------- */
// CPU reference: identical update and stopping rule, single-threaded.

int solve_cpu(float* u, int X, int Y, const Options& o, double* seconds)
{
  size_t n = (size_t)X * Y;
  float* unew = (float*)malloc(n * sizeof(float));
  memcpy(unew, u, n * sizeof(float));
  clock_t t0 = clock();
  int it = 0;
  while (it < o.maxItr) {
    it++;
    for (int y = 1; y < Y - 1; y++)
      for (int x = 1; x < X - 1; x++)
        unew[x + y * X] = 0.25f * (u[x + 1 + y * X] + u[x - 1 + y * X] + u[x + (y + 1) * X] + u[x + (y - 1) * X]);
    bool check = (it % o.checkEvery) == 0;
    float maxerr = 0.0f;
    if (check)
      for (int y = 1; y < Y - 1; y++)
        for (int x = 1; x < X - 1; x++) maxerr = fmaxf(maxerr, fabsf(unew[x + y * X] - u[x + y * X]));
    float* tmp = u;  // swap (copy back at the end so the caller's buffer holds the result)
    u = unew;
    unew = tmp;
    if (check && maxerr < o.tol) break;
  }
  *seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;
  if (it % 2 == 1) {  // result currently lives in the malloc'd buffer: copy it into the caller's array
    memcpy(unew, u, n * sizeof(float));
    free(u);
  } else {
    free(unew);
  }
  return it;
}

/* --------------------------------------------------------------------------------------------------------------- */

int solve_gpu(float* u_h, int X, int Y, const Options& o, float* ms)
{
  size_t size = (size_t)X * Y * sizeof(float);
  float *unew_d, *uold_d, *udiff_d;
  CUDA_CHECK(cudaMalloc(&uold_d, size));
  CUDA_CHECK(cudaMalloc(&unew_d, size));
  CUDA_CHECK(cudaMalloc(&udiff_d, (size_t)(X - 2) * (Y - 2) * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(uold_d, u_h, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(unew_d, u_h, size, cudaMemcpyHostToDevice));
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 dimGrid((X - 2) / BLOCK_SIZE_X, (Y - 2) / BLOCK_SIZE_Y);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  int nsteps = 0;
  float maxerr = INFINITY;
  while (nsteps < o.maxItr) {
    nsteps++;
    Jacobi<<<dimGrid, dimBlock>>>(uold_d, unew_d, X, Y);
    if (nsteps % o.checkEvery == 0) {
      udiff<<<dimGrid, dimBlock>>>(udiff_d, uold_d, unew_d, X, Y);
      int index_max;  // 1-based index of the largest |difference|
      CUBLAS_CHECK(cublasIsamax(handle, (X - 2) * (Y - 2), udiff_d, 1, &index_max));
      CUDA_CHECK(cudaMemcpy(&maxerr, &udiff_d[index_max - 1], sizeof(float), cudaMemcpyDeviceToHost));
      maxerr = fabsf(maxerr);
    }
    float* tmp = uold_d;  // newest iterate is now in uold_d
    uold_d = unew_d;
    unew_d = tmp;
    if (o.dumpEvery && nsteps % o.dumpEvery == 0) {
      CUDA_CHECK(cudaMemcpy(u_h, uold_d, size, cudaMemcpyDeviceToHost));
      output(u_h, X, Y, nsteps);
    }
    if (maxerr < o.tol) break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(ms, start, stop));
  CUDA_CHECK(cudaMemcpy(u_h, uold_d, size, cudaMemcpyDeviceToHost));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  cudaFree(uold_d);
  cudaFree(unew_d);
  cudaFree(udiff_d);
  return maxerr < o.tol ? nsteps : -nsteps;
}

/* --------------------------------------------------------------------------------------------------------------- */

Options parse_args(int argc, char** argv)
{
  Options o;
  for (int i = 1; i < argc; i++) {
    const char* a = argv[i];
    bool more = i + 1 < argc;
    if (!strcmp(a, "--nx") && more) o.xdim = atoi(argv[++i]);
    else if (!strcmp(a, "--ny") && more) o.ydim = atoi(argv[++i]);
    else if (!strcmp(a, "--tol") && more) o.tol = (float)atof(argv[++i]);
    else if (!strcmp(a, "--max-iter") && more) o.maxItr = atoi(argv[++i]);
    else if (!strcmp(a, "--check-every") && more) o.checkEvery = atoi(argv[++i]);
    else if (!strcmp(a, "--dump-every") && more) o.dumpEvery = atoi(argv[++i]);
    else if (!strcmp(a, "--cpu")) o.cpu = true;
    else {
      fprintf(stderr, "Unknown or incomplete option: %s\n", a);
      exit(2);
    }
  }
  if (o.xdim % BLOCK_SIZE_X || o.ydim % BLOCK_SIZE_Y || o.xdim <= 0 || o.ydim <= 0) {
    fprintf(stderr, "--nx and --ny must be positive multiples of %d and %d\n", BLOCK_SIZE_X, BLOCK_SIZE_Y);
    exit(2);
  }
  if (o.checkEvery < 1) o.checkEvery = 1;
  return o;
}

int main(int argc, char** argv)
{
  Options o = parse_args(argc, argv);
  int X = o.xdim + 2, Y = o.ydim + 2;  // add the boundary layer
  size_t n = (size_t)X * Y;
  float* u_h = (float*)calloc(n, sizeof(float));
  initialize(u_h, X, Y);
  if (o.dumpEvery) output(u_h, X, Y, 0);

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("Grid %d x %d interior points on %s, tol %.1e, convergence check every %d iterations\n", o.xdim, o.ydim,
         prop.name, o.tol, o.checkEvery);

  float* u_gpu = (float*)malloc(n * sizeof(float));
  memcpy(u_gpu, u_h, n * sizeof(float));
  float ms = 0.0f;
  int itGpu = solve_gpu(u_gpu, X, Y, o, &ms);
  double updates = (double)o.xdim * o.ydim * abs(itGpu);
  printf("GPU: %s after %d iterations in %.1f ms  (%.0f million cell updates/s)\n",
         itGpu > 0 ? "converged" : "NOT converged", abs(itGpu), ms, updates / (ms * 1e3));
  output(u_gpu, X, Y, abs(itGpu));

  if (o.cpu) {
    float* u_cpu = (float*)malloc(n * sizeof(float));
    memcpy(u_cpu, u_h, n * sizeof(float));
    double s = 0.0;
    int itCpu = solve_cpu(u_cpu, X, Y, o, &s);
    double maxdiff = 0.0;
    for (size_t k = 0; k < n; k++) maxdiff = fmax(maxdiff, fabs((double)u_cpu[k] - u_gpu[k]));
    printf("CPU: %d iterations in %.1f ms  (%.0f million cell updates/s)\n", itCpu, s * 1e3,
           (double)o.xdim * o.ydim * itCpu / (s * 1e6));
    printf("Speedup %.1fx, max |GPU - CPU| = %.2e\n", s * 1e3 / ms, maxdiff);
    free(u_cpu);
  }
  free(u_gpu);
  free(u_h);
  return itGpu > 0 ? 0 : 1;
}
