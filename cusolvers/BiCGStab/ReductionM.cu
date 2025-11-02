#include "ReductionM.h"
#include <cooperative_groups/reduce.h>
#include "globals.h"

namespace cusolver
{
    namespace cg = cooperative_groups;

    __device__ void extra_action__(double res, ExtraAction action)
    {
        switch (action)
        {
        case ExtraAction::compute_rs_new_and_beta:
            rs_new = res;
            beta = (rs_new / rs_old) * (alpha / omega);
            rs_old = rs_new;
            break;
        case ExtraAction::compute_alpha:
            alpha = rs_new / res;
            break;
        case ExtraAction::compute_omega:
            omega = buffer / res;
            break;
        case ExtraAction::compute_buffer:
            buffer = res;
            break;
        case ExtraAction::compute_rs_old:
            rs_old = res;
            break;
        default:
            break;
        }
    }
    template <class T, unsigned int blockSize>
    __global__ void reduce5(T* g_idata, T* second, T* g_odata, unsigned int n, bool first, bool last, ExtraAction action) {
        // Handle to thread block group
        cg::thread_block cta = cg::this_thread_block();
        extern __shared__ double sdata[];

        // perform first level of reduction,
        // reading from global memory, writing to shared memory
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

        T mySum = (i < n) ? g_idata[i] * (first ? second[i] : 1) : 0;
        if (i + blockSize < n) mySum += g_idata[i + blockSize] * (first ? second[i + blockSize] : 1);

        sdata[tid] = mySum;
        cg::sync(cta);

        // do reduction in shared mem
        if ((blockSize >= 512) && (tid < 256)) {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        cg::sync(cta);

        if ((blockSize >= 256) && (tid < 128)) {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        cg::sync(cta);

        if ((blockSize >= 128) && (tid < 64)) {
            sdata[tid] = mySum = mySum + sdata[tid + 64];
        }

        cg::sync(cta);

        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        if (cta.thread_rank() < 32) {
            // Fetch final intermediate sum from 2nd warp
            if (blockSize >= 64) mySum += sdata[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
                mySum += tile32.shfl_down(mySum, offset);
            }
        }

        // write result for this block to global mem
        if (cta.thread_rank() == 0)
        {
            g_odata[blockIdx.x] = mySum;
            if (last)
            {
                extra_action__(mySum, action);
            }
        }
    }

    CudaReductionM::CudaReductionM() {}
    CudaReductionM::CudaReductionM(unsigned int N, unsigned int thr)
    {
        bool doubleRead = false;
        if (thr < 64)
        {
            std::cout << "more threads needed " << std::endl;
            threads = 64;
        }

        unsigned int temp_ = N;
        threads = thr;
        N_v.push_back(N);

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

        if (res_array != nullptr) cudaFree(res_array);
        cudaMalloc((void**)&res_array, sizeof(double) * N_v[1]);

        if (arr != nullptr) delete[] arr;
        arr = new double* [steps + 1];
    }
    CudaReductionM::~CudaReductionM()
    {
        cudaFree(res_array); res_array = nullptr;
        delete[] arr; 	arr = nullptr;
        grid_v.clear();
        N_v.clear();
    }
    double CudaReductionM::reduce(double* v1, double* v2, bool withCopy, ExtraAction action)
    {
        arr[0] = v1;	second = v2;
        for (unsigned int i = 1; i <= steps; i++)
            arr[i] = res_array;

        switch (threads)
        {
        case(512):
            for (unsigned int i = 0; i < steps; i++)	reduce5<double, 512> << <grid_v[i], threads, smem >> > (arr[i], second, arr[i + 1], N_v[i], i == 0, i == steps - 1, action);
            break;
        case(256):
            for (unsigned int i = 0; i < steps; i++)	reduce5<double, 256> << <grid_v[i], threads, smem >> > (arr[i], second, arr[i + 1], N_v[i], i == 0, i == steps - 1, action);
            break;
        case(128):
            for (unsigned int i = 0; i < steps; i++)	reduce5<double, 128> << <grid_v[i], threads, smem >> > (arr[i], second, arr[i + 1], N_v[i], i == 0, i == steps - 1, action);
            break;
        case(64):
            for (unsigned int i = 0; i < steps; i++)	reduce5<double, 64> << <grid_v[i], threads, smem >> > (arr[i], second, arr[i + 1], N_v[i], i == 0, i == steps - 1, action);
            break;
        default:
            break;
        }
        //for (unsigned int i = 0; i < steps; i++)	dot_product << < grid_v[i], threads, smem >> > (arr[i], second, N_v[i], arr[i + 1], i == 0, i == steps - 1, action);

        if (withCopy) cudaMemcpy(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);

        return res;
    }
    CuGraph CudaReductionM::make_graph(double* v1, double* v2, bool withCopy, ExtraAction action)
    {
        arr[0] = v1;	second = v2;
        for (unsigned int i = 1; i <= steps; i++)
            arr[i] = res_array;

        void* kernel;

        if (threads == 512) kernel = reinterpret_cast<void*>(&reduce5<double, 512>);
        if (threads == 256) kernel = reinterpret_cast<void*>(&reduce5<double, 256>);
        if (threads == 128) kernel = reinterpret_cast<void*>(&reduce5<double, 128>);
        if (threads == 64)  kernel = reinterpret_cast<void*>(&reduce5<double, 64>);

        CuGraph graph;
        for (unsigned int i = 0; i < steps; i++)
        {
            bool first = (i == 0);
            bool last = (i == steps - 1);
            void* args[] = { &arr[i], &second, &arr[i + 1], &N_v[i], &first, &last, &action };
            graph.add_kernel_node(threads, grid_v[i], kernel, args, smem);
            //dangling pointers?
        }
        if (withCopy) graph.add_copy_node(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);
        return graph;
    }

}