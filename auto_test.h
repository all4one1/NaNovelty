#pragma once
#include <iostream>
#include "cuda_runtime.h"




void auto_test()
{
	// auto-test example: 

	int n = 6;	// rank of a square matrix 
	int nval = 24;	// number of non-zero elements of a matrix

	// Example of a sparse matrix
	double sparse_matrix_elements[24] = { 30, 3, 4, 4, 22, 1, 3, 5, 7, 33, 6, 7, 1, 2, 42, 3, 3, 2, 11, 52, 2, 3, 9, 26 };
	int column[24] = { 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5 };
	int row[7] = { 0, 3, 7, 12, 17, 21, 24 };

	//CuCG::SparseMatrixCuda SMC(n, nval, sparse_matrix_elements, column, row);

	double f_host[6] = { 0, 0, 0, 0, 0, 0 }; //init values
	double b_host[6] = { 1, 2, 3, 3, 2, 1 };
	double* f_dev, * f0_dev, *b_dev;
	cudaMalloc((void**)&f_dev, sizeof(double) * n);
	cudaMalloc((void**)&f0_dev, sizeof(double) * n);
	cudaMalloc((void**)&b_dev, sizeof(double) * n);

	cudaMemcpy(f0_dev, f_host, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(f_dev, f_host, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(b_dev, b_host, sizeof(double) * n, cudaMemcpyHostToDevice);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	cusolver::CudaLaunchSetup launch(n);
	#define KERNEL1D launch.grid1d, launch.block1d

	cusolver::SparseMatrixData* sm_dev = nullptr;
	cusolver::allocate_on_device(&sm_dev, n, nval, sparse_matrix_elements, column, row);
	cusolver::solve_jacobi(n, sm_dev, f_dev, f0_dev, b_dev);



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// check:
	cudaMemcpy(f_host, f_dev, sizeof(double) * n, cudaMemcpyDeviceToHost);
	std::cout << "cuda test: ";	for (int i = 0; i < n; i++)		std::cout << f_host[i] << " ";	 std::cout << std::endl;
	double cg[6] = { 0.1826929218e-1,	0.7636750835e-1,	0.5570467736e-1,	0.6371099009e-1,	0.2193724104e-1,	0.2351661001e-1 };
	std::cout << "solution : ";	for (int i = 0; i < n; i++)		std::cout << cg[i] << " ";	std::cout << std::endl;

	cudaFree(f_dev);
	cudaFree(f0_dev);
	cudaFree(b_dev);
}