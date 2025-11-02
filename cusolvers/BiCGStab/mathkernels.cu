#include "mathkernels.h"
#include "globals.h"


namespace cusolver
{
    __global__ void hadamard_product(double* res, double* v1, double* v2, unsigned int N)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            res[i] = v1[i] * v2[i];
        }
    }

    __global__ void matrix_dot_vector(double* res, SparseMatrixData *M, double* vm, unsigned int N)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            double s = 0;
            for (int j = M->row[i]; j < M->row[i + 1]; j++)
            {
                s += M->val[j] * vm[M->col[j]];
            }
            res[i] = s;
        }
    }

    __global__ void vector_add_vector(double* res, double* v1, double* v2, unsigned int N)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            res[i] = v1[i] + v2[i];
        }
    }

    __global__ void vector_substract_vector(double* res, double* v1, double* v2, unsigned int N)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            res[i] = v1[i] - v2[i];
        }
    }

    __global__ void scalar_dot_vector(double* res, double scalar, double* v, unsigned int N)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            res[i] = scalar * v[i];
        }
    }

    __global__ void vector_minus_matrix_dot_vector(double* res, double* v, SparseMatrixData *M, double* vm, unsigned int N)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            double s = 0;
            for (int j = M->row[i]; j < M->row[i + 1]; j++)
            {
                s += M->val[j] * vm[M->col[j]];
            }
            res[i] = v[i] - s;
        }
    }

    __global__ void vector_minus_vector(double* res, double* v1, double* v2, unsigned int N, KernelCoefficient choice)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            double coef = 1;
            switch (choice)
            {
            case KernelCoefficient::alpha:
                coef = alpha;
                break;
            case KernelCoefficient::omega:
                coef = omega;
                break;
            default:
                break;
            }
            res[i] = v1[i] - coef * v2[i];
        }
    }

    __global__ void vector_add_2vectors(double* res, double* v, double* vs1, double* vs2, unsigned int N, KernelCoefficient choice)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            double coef1 = 1, coef2 = 1;
            switch (choice)
            {
            case KernelCoefficient::beta_and_omega:
                coef1 = beta;
                coef2 = -beta * omega;
                break;
            case KernelCoefficient::alpha_and_omega:
                coef1 = alpha;
                coef2 = omega;
                break;
            default:
                break;
            }
            res[i] = v[i] + coef1 * vs1[i] + coef2 * vs2[i];
        }
    }

    __global__ void vector_set_to_vector(double* res, double* v, unsigned int N)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            res[i] = v[i];
        }
    }

    __global__ void vector_set_to_value(double* res, double scalar, unsigned int N)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            res[i] = scalar;
        }
    }
}