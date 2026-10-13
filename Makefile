NVCC ?= nvcc
laplace: laplace.cu
	$(NVCC) -O3 $< -o $@ -lcublas
clean:
	rm -f laplace laplace.exe solution_iter_*.txt
