#pragma once
#include "../commontools.h"
#include "ReductionM.h"

namespace cusolver
{
	//struct _BICGSTAB {double rs_old, rs_new, alpha, beta, buffer, buffer2, omega;};

	struct BiCGSTAB
	{
		double* r = nullptr, * r_hat = nullptr, * p = nullptr, * t = nullptr, * s = nullptr, * v = nullptr;

		double eps = 1e-10;
		double rs_host = 1;
		unsigned int N = 0, Nbytes = 0, k = 0;
		unsigned int threads = 1, blocks = 1;
		CuGraph graph;
		CudaReductionM* CR;

		BiCGSTAB::BiCGSTAB();
		BiCGSTAB::BiCGSTAB(unsigned int N_, double* x, double* x0, double* b,
			SparseMatrixData* A, CudaLaunchSetup kernel_setting, unsigned int reduction_threads = 256);
		void BiCGSTAB::solve_directly(double* x, double* x0, double* b, SparseMatrixData* A);
		void BiCGSTAB::make_graph(double* x, double* x0, double* b, SparseMatrixData* A);
		void BiCGSTAB::solve_with_graph(double* x, double* x0, double* b, SparseMatrixData* A);
	};
}