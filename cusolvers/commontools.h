#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
//#include "../CudaReduction/CuReduction.h"
#include <iostream>
#include <vector>

namespace cusolver
{
    #define KERNEL(func) func<<< blocks, threads>>>

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

    struct CuGraph
    {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graphExec = nullptr;
        std::vector<cudaGraphNode_t> Nodes;

        CuGraph();
        ~CuGraph();

        void add_kernel_node(unsigned int threads_per_block, unsigned int num_blocks, void* kernel, void** args,
            unsigned int sbytes = 0, bool depend_on_previous = true);

        void add_copy_node(void* dst, const void* src, unsigned int Nbytes, cudaMemcpyKind direction, bool depend_on_previous = true);

        void add_graph_as_node(cudaGraph_t& childGraph, bool depend_on_previous = true);
        void add_graph_as_node(CuGraph& childGraph, bool depend_on_previous = true);

        void instantiate();
        void launch();
    };

    struct CudaReduction
    {
        std::vector<unsigned int> grid_v;
        std::vector<unsigned int> N_v;

        #define def_threads 512
        unsigned int N = 0;
        unsigned int steps = 0, threads = def_threads, smem = sizeof(double) * def_threads;

        double* res_array = nullptr;
        double res = 0;
        double** arr = nullptr;

        void* reduction_kernel;
        enum ReductionType { ABSSUM, SIGNEDSUM, DOTPRODUCT } type = ABSSUM;

        CudaReduction(double* device_ptr, unsigned int N, unsigned int thr = def_threads);
        CudaReduction(unsigned int N, unsigned int thr = 1024);
        CudaReduction();
        ~CudaReduction();
        void set_reduced_size(unsigned int N, unsigned int thr, bool doubleRead);


        void print_check();
        double reduce(bool withCopy = true);
        double reduce_legacy(bool withCopy = true);
        double reduce(double* device_ptr, bool withCopy = true);
        static double reduce(double* device_ptr, unsigned int N, unsigned int thr = def_threads, bool withCopy = true);

        double check_on_cpu(double* device_ptr, unsigned int N);

        void auto_test();

        CuGraph CudaReduction::make_graph(double* device_ptr, bool withCopy);
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

    void allocate_sparse_matrix_on_device(SparseMatrixData** sm_dev, int n_, int nnz_,
        double* val_host = nullptr, int* col_host = nullptr, int* row_host = nullptr);

    void copy_sparse_matrix_to_device(SparseMatrixData* sm_dev, int nnz, int n,
        double* val_host, int* col_host, int* row_host);

    void update_matrix_values_on_device(SparseMatrixData* sm_dev, int nnz, int n, double* val_host);

}