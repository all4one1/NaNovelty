#include "jacobi.h"

namespace cusolver
{

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
            solve_jacobi_step << <KERNEL1D >> > (M_dev, f_dev, f0_dev, b_dev);

            res = CR.reduce(f_dev, true);
            eps = abs(res - res0) / (res0 + 1e-5);
            res0 = res;

            std::swap(f_dev, f0_dev);

            if (eps < eps_iter)	break;
        }
    }

    __global__ void cusolver::solve_jacobi_step(SparseMatrixData* M, double* f, double* f0, double* b)
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
}