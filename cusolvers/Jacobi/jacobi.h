#pragma once
#include "../commontools.h"
namespace cusolver
{
    void solve_jacobi(int n, SparseMatrixData* M_dev, double* f_dev, double* f0_dev, double* b_dev);

    __global__ void solve_jacobi_step(SparseMatrixData* M, double* f, double* f0, double* b);
}