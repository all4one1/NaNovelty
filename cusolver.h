#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaReduction/CuReduction.h"
#include <iostream>

namespace cusolver
{
    struct GPU_
    {
        double mem_start = 0;
        int devID = 0;
        GPU_(int id = 0);
        double get_gpu_memory_used();
        void show_memory_usage_MB();
    };

    struct CudaLaunchSetup
    {
        dim3 grid3d, block3d, grid2d, block2d, grid1d, block1d;
        unsigned int threads3d = 8, threads2d = 32, threads = 1024;
        CudaLaunchSetup(unsigned int N, unsigned int nx = 1, unsigned int ny = 1, unsigned nz = 1);
    };

    struct SparseMatrixData
    {
        double* val = nullptr;
        int* col = nullptr;
        int* row = nullptr;
        int n = 0;
        int nnz = 0;
        int nrow = 0;
    };

    // Объявления функций
    void allocate_sparse_matrix_on_device(SparseMatrixData** sm_dev, int n_, int nnz_,
        double* val_host = nullptr, int* col_host = nullptr, int* row_host = nullptr);

    void copy_sparse_matrix_to_device(SparseMatrixData* sm_dev, int nnz, int n,
        double* val_host, int* col_host, int* row_host);

    void update_matrix_values_on_device(SparseMatrixData* sm_dev, int nnz, int n, double* val_host);

    void solve_jacobi(int n, SparseMatrixData* M_dev, double* f_dev, double* f0_dev, double* b_dev);

    // Объявление ядра CUDA
    __global__ void solve_jacobi_step(SparseMatrixData* M, double* f, double* f0, double* b);
}