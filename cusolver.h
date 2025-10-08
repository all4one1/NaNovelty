#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaReduction/CuReduction.h"
#include <iostream>

#define cudaCheckError {cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) {printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));  exit(0);}}

namespace cusolver
{
    struct GPU_
    {
        double mem_start = 0;
        int devID = 0;
        GPU_(int id = 0)
        {
            devID = id;
            cudaSetDevice(devID);
            mem_start = get_gpu_memory_used();
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, devID);
            printf("\nDevice %d: \"%s\"\n", devID, deviceProp.name);
        }
        double get_gpu_memory_used()
        {
            size_t free_byte, total_byte;
            cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

            if (cudaSuccess != cuda_status)
            {
                printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
                return -1;
            }
            double free_mb = (double)free_byte / 1024.0 / 1024.0;
            double total_mb = (double)total_byte / 1024.0 / 1024.0;
            double used_mb = total_mb - free_mb;
            return used_mb;
        }
        void show_memory_usage_MB()
        {
            std::cout << "Approximately GPU memory allocated: " << get_gpu_memory_used() - mem_start << " MB" << std::endl;
        }
    };

    struct CudaLaunchSetup
    {
        dim3 grid3d, block3d, grid2d, block2d, grid1d, block1d;
        unsigned int threads3d = 8, threads2d = 32, threads = 1024;

        CudaLaunchSetup(unsigned int N, unsigned int nx = 1, unsigned int ny = 1, unsigned nz = 1)
        {
            grid3d = dim3(
                (unsigned int)ceil((nx + 1.0) / threads3d),
                (unsigned int)ceil((ny + 1.0) / threads3d),
                (unsigned int)ceil((nz + 1.0) / threads3d));
            block3d = dim3(threads3d, threads3d, threads3d);

            grid2d = dim3(
                (unsigned int)ceil((nx + 1.0) / threads2d),
                (unsigned int)ceil((ny + 1.0) / threads2d));
            block2d = dim3(threads2d, threads2d);

            grid1d = dim3((unsigned int)ceil((N + 0.0) / threads));
            block1d = threads;
        };
    };

    struct SparseMatrixData
    {
        /*		Compressed Sparse Row Format	 */
        double* val = nullptr;
        int* col = nullptr;
        int* row = nullptr;
        int n = 0;	// size of the matrix (n x n)
        int nnz = 0;	// number of non-zero 
        int nrow = 0; // size of row array (n + 1)
    };

    void allocate_sparse_matrix_on_device(SparseMatrixData** sm_dev, int n_, int nnz_,
        double* val_host = nullptr, int* col_host = nullptr, int* row_host = nullptr)
    {
        SparseMatrixData temp_host;
        temp_host.n = n_;
        temp_host.nnz = nnz_;
        temp_host.nrow = n_ + 1;

        if (*sm_dev == nullptr)
        {
            cudaMalloc((void**)&(*sm_dev), sizeof(SparseMatrixData));

            cudaMalloc((void**)&temp_host.val, sizeof(double) * temp_host.nnz);
            cudaMalloc((void**)&temp_host.col, sizeof(int) * temp_host.nnz);
            cudaMalloc((void**)&temp_host.row, sizeof(int) * temp_host.nrow);
            cudaMemcpy(*sm_dev, &temp_host, sizeof(SparseMatrixData), cudaMemcpyHostToDevice);

            if (val_host != nullptr) cudaMemcpy(temp_host.val, val_host, sizeof(double) * temp_host.nnz, cudaMemcpyHostToDevice);
            if (col_host != nullptr) cudaMemcpy(temp_host.col, col_host, sizeof(int) * temp_host.nnz, cudaMemcpyHostToDevice);
            if (row_host != nullptr) cudaMemcpy(temp_host.row, row_host, sizeof(int) * temp_host.nrow, cudaMemcpyHostToDevice);
        }
        else
        {
            std::cout << "warning: allocate_on_device" << std::endl;
        }
    }
    void copy_sparse_matrix_to_device(SparseMatrixData* sm_dev, int nnz, int n, double* val_host, int* col_host, int* row_host)
    {
        SparseMatrixData temp_host;
        cudaMemcpy(&temp_host, sm_dev, sizeof(SparseMatrixData), cudaMemcpyDeviceToHost);

        cudaMemcpy(temp_host.val, val_host, sizeof(double) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(temp_host.col, col_host, sizeof(int) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(temp_host.row, row_host, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
    }
    void update_matrix_values_on_device(SparseMatrixData* sm_dev, int nnz, int n, double* val_host)
    {
        SparseMatrixData temp_host;
        cudaMemcpy(&temp_host, sm_dev, sizeof(SparseMatrixData), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_host.val, val_host, sizeof(double) * nnz, cudaMemcpyHostToDevice);
    }

    __global__ void solve_jacobi_step(SparseMatrixData* M, double* f, double* f0, double* b)
    {
        unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

        double sum, diag;

        auto line = [&](int q, double& sum, double& diag)
        {
            sum = 0.0;
            diag = 1.0;
            for (int j = M->row[q]; j < M->row[q + 1]; j++)
            {
                sum += M->val[j] * f0[M->col[j]];
                if (M->col[j] == q) diag = M->val[j];
            }
        };

        if (i < M->n)
        {
            line(i, sum, diag);
            f[i] = f0[i] + (b[i] - sum) / diag;
        }
    }

    void solve_jacobi(int n, SparseMatrixData* M_dev, double* f_dev, double* f0_dev, double* b_dev)
    {
        double eps_iter = 1e-5;
        double eps = 0, res = 0, res0 = 0;
        unsigned int k;

        CudaReduction CR(f_dev, n, 512);
        CudaLaunchSetup launch(n);
        #define KERNEL1D launch.grid1d, launch.block1d

        for (k = 1; k < 1000000; k++)
        {
            solve_jacobi_step <<<KERNEL1D >>> (M_dev, f_dev, f0_dev, b_dev);

            res = CR.reduce(f_dev, true);
            eps = abs(res - res0) / (res0 + 1e-5);
            res0 = res;

            std::swap(f_dev, f0_dev);

            if (eps < eps_iter)	break;
        }
        //std::cout << "iterations, k = " << k << std::endl;
    }
}

