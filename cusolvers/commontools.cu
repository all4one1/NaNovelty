#include "commontools.h"
#include <cooperative_groups/reduce.h>

#define cudaCheckError {cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) {printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));  exit(0);}}

namespace cusolver
{
    GPU_::GPU_(int id)
    {
        devID = id;
        cudaSetDevice(devID);
        mem_start = get_gpu_memory_used();
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, devID);
        printf("\nDevice %d: \"%s\"\n", devID, deviceProp.name);
    }

    double GPU_::get_gpu_memory_used()
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

    void GPU_::show_memory_usage_MB()
    {
        std::cout << "Approximately GPU memory allocated: " << get_gpu_memory_used() - mem_start << " MB" << std::endl;
    }

    CudaLaunchSetup::CudaLaunchSetup(unsigned int N, unsigned int nx, unsigned int ny, unsigned nz)
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
    }

    CuGraph::CuGraph()
    {
        cudaGraphCreate(&graph, 0);
    }
    CuGraph::~CuGraph()
    {
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
        graph = nullptr;
        graphExec = nullptr;
        Nodes.clear();
    }
    void CuGraph::add_kernel_node(unsigned int threads_per_block, unsigned int num_blocks, void* kernel, void** args,
        unsigned int sbytes, bool depend_on_previous)
    {
        Nodes.emplace_back();

        cudaKernelNodeParams param = { 0 };
        param.blockDim = threads_per_block;
        param.gridDim = num_blocks;
        param.func = kernel;
        param.kernelParams = args;
        param.sharedMemBytes = sbytes;

        size_t current = Nodes.size() - 1;
        if (depend_on_previous && current > 0)
            cudaGraphAddKernelNode(&Nodes.back(), graph, &Nodes[current - 1], 1, &param);
        else
            cudaGraphAddKernelNode(&Nodes.back(), graph, nullptr, 0, &param);
    }
    void CuGraph::add_copy_node(void* dst, const void* src, unsigned int Nbytes, cudaMemcpyKind direction, bool depend_on_previous)
    {
        Nodes.emplace_back();
        size_t current = Nodes.size() - 1;
        if (depend_on_previous && current > 0)
            cudaGraphAddMemcpyNode1D(&Nodes[current], graph, &Nodes[current - 1], 1, dst, src, Nbytes, direction);
        else
            cudaGraphAddMemcpyNode1D(&Nodes[current], graph, nullptr, 0, dst, src, Nbytes, direction);
    }
    void CuGraph::add_graph_as_node(cudaGraph_t& childGraph, bool depend_on_previous)
    {
        Nodes.emplace_back();
        size_t current = Nodes.size() - 1;
        if (depend_on_previous && current > 0)
            cudaGraphAddChildGraphNode(&Nodes[current], graph, &Nodes[current - 1], 1, childGraph);
        else
            cudaGraphAddChildGraphNode(&Nodes[current], graph, nullptr, 0, childGraph);
    }
    void CuGraph::add_graph_as_node(CuGraph& childGraph, bool depend_on_previous)
    {
        Nodes.emplace_back();
        size_t current = Nodes.size() - 1;
        if (depend_on_previous && current > 0)
            cudaGraphAddChildGraphNode(&Nodes[current], graph, &Nodes[current - 1], 1, childGraph.graph);
        else
            cudaGraphAddChildGraphNode(&Nodes[current], graph, nullptr, 0, childGraph.graph);
    }
    void CuGraph::instantiate()
    {
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    }
    void CuGraph::launch()
    {
        cudaGraphLaunch(graphExec, 0);
    }

    template <class T, unsigned int blockSize>
    __global__ void reduce5abs(T* g_idata, T* g_odata, unsigned int n) {
        // Handle to thread block group
        cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
        extern __shared__ T sdata[];

        // perform first level of reduction,
        // reading from global memory, writing to shared memory
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

        T mySum = (i < n) ? abs(g_idata[i]) : 0;

        if (i + blockSize < n) mySum += abs(g_idata[i + blockSize]);

        sdata[tid] = mySum;
        cooperative_groups::sync(cta);

        // do reduction in shared mem
        if ((blockSize >= 512) && (tid < 256)) {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        cooperative_groups::sync(cta);

        if ((blockSize >= 256) && (tid < 128)) {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        cooperative_groups::sync(cta);

        if ((blockSize >= 128) && (tid < 64)) {
            sdata[tid] = mySum = mySum + sdata[tid + 64];
        }

        cooperative_groups::sync(cta);

        cooperative_groups::thread_block_tile<32> tile32 = cooperative_groups::tiled_partition<32>(cta);

        if (cta.thread_rank() < 32) {
            // Fetch final intermediate sum from 2nd warp
            if (blockSize >= 64) mySum += sdata[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
                mySum += tile32.shfl_down(mySum, offset);
            }
        }

        // write result for this block to global mem
        if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
    }

    CudaReduction::CudaReduction()
    {

    }
    CudaReduction::CudaReduction(unsigned int N, unsigned int thr)
    {
        set_reduced_size(N, thr, true);

        if (res_array != nullptr) cudaFree(res_array);
        cudaMalloc((void**)&res_array, sizeof(double) * N_v[1]);

        if (arr != nullptr) delete[] arr;
        arr = new double* [steps + 1];
    }
    CudaReduction::CudaReduction(double* device_ptr, unsigned int N, unsigned int thr)
    {
        set_reduced_size(N, thr, true);

        if (res_array != nullptr) cudaFree(res_array);
        cudaMalloc((void**)&res_array, sizeof(double) * N_v[1]);

        if (arr != nullptr) delete[] arr;
        arr = new double* [steps + 1];

        arr[0] = device_ptr;
        for (unsigned int i = 1; i <= steps; i++)
            arr[i] = res_array;
    }
    CudaReduction::~CudaReduction()
    {
        cudaFree(res_array); res_array = nullptr;
        delete[] arr; arr = nullptr;
        grid_v.clear();
        N_v.clear();
    }
    void CudaReduction::set_reduced_size(unsigned int N, unsigned int thr, bool doubleRead)
    {
        if (thr < 64)
        {
            std::cout << "more threads needed " << std::endl;
            threads = 64;
        }

        unsigned int temp_ = N;
        threads = thr;
        N_v.push_back(N);

        if (threads == 512) reduction_kernel = reinterpret_cast<void*>(&reduce5abs<double, 512>);
        if (threads == 256) reduction_kernel = reinterpret_cast<void*>(&reduce5abs<double, 256>);
        if (threads == 128) reduction_kernel = reinterpret_cast<void*>(&reduce5abs<double, 128>);
        if (threads == 64)  reduction_kernel = reinterpret_cast<void*>(&reduce5abs<double, 64>);


        steps = 0;
        while (true)
        {
            steps++;
            if (doubleRead) temp_ = (temp_ + (threads * 2 - 1)) / (threads * 2);
            else temp_ = (temp_ + threads - 1) / threads;

            grid_v.push_back(temp_);
            N_v.push_back(temp_);
            if (temp_ == 1)  break;
        }
    }
    double CudaReduction::reduce(double* device_ptr, bool withCopy)
    {
        arr[0] = device_ptr;
        for (unsigned int i = 1; i <= steps; i++)
            arr[i] = res_array;

        return reduce(withCopy);
    }
    double CudaReduction::reduce(double* device_ptr, unsigned int N, unsigned int thr, bool withCopy)
    {
        CudaReduction temp(device_ptr, N, thr);
        return temp.reduce(withCopy);
    }
    double CudaReduction::reduce(bool withCopy)
    {
        //for (unsigned int i = 0; i < steps; i++) reduce_<double>(N_v[i], threads, grid_v[i], 5, arr[i], arr[i + 1]);

        //for (unsigned int i = 0; i < steps; i++)
        //{
        //	void* args[] = { &arr[i], &arr[i + 1], &N_v[i] };
        //	cudaLaunchKernel(reduction_kernel, grid_v[i], threads, args, smem, 0);
        //}

        if (threads == 512) for (unsigned int i = 0; i < steps; i++) reduce5abs<double, 512> << < grid_v[i], threads, smem >> > (arr[i], arr[i + 1], N_v[i]);
        if (threads == 256) for (unsigned int i = 0; i < steps; i++) reduce5abs<double, 256> << < grid_v[i], threads, smem >> > (arr[i], arr[i + 1], N_v[i]);
        if (threads == 128) for (unsigned int i = 0; i < steps; i++) reduce5abs<double, 128> << < grid_v[i], threads, smem >> > (arr[i], arr[i + 1], N_v[i]);
        if (threads == 64)  for (unsigned int i = 0; i < steps; i++) reduce5abs<double, 64> << < grid_v[i], threads, smem >> > (arr[i], arr[i + 1], N_v[i]);

        if (withCopy) cudaMemcpy(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);
        return res;
    }
    CuGraph CudaReduction::make_graph(double* device_ptr, bool withCopy)
    {
        arr[0] = device_ptr;
        for (unsigned int i = 1; i <= steps; i++)
            arr[i] = res_array;

        CuGraph graph;
        for (unsigned int i = 0; i < steps; i++)
        {
            void* args[3] = { &arr[i], &N_v[i], &arr[i + 1] };        
            void* kernel;
            if (threads == 512) for (unsigned int i = 0; i < steps; i++) kernel = reduce5abs<double, 512>;
            if (threads == 256) for (unsigned int i = 0; i < steps; i++) kernel = reduce5abs<double, 256>;
            if (threads == 128) for (unsigned int i = 0; i < steps; i++) kernel = reduce5abs<double, 128>;
            if (threads == 64)  for (unsigned int i = 0; i < steps; i++) kernel = reduce5abs<double, 64>;

            graph.add_kernel_node(threads, grid_v[i], kernel, args, smem);
        }



        if (withCopy) graph.add_copy_node(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);

        return graph;
    }




    void allocate_sparse_matrix_on_device(SparseMatrixData** sm_dev, int n_, int nnz_,
        double* val_host, int* col_host, int* row_host)
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


}