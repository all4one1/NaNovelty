#include "bicgstab.h"
#include "mathkernels.h"
//#include "globals.h"

namespace cusolver
{
	__device__ double rs_old = 1, rs_new = 1, alpha = 1, beta = 1, buffer = 1, buffer2 = 1, omega = 1;
	//__device__ _BICGSTAB _bicgstab{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
	#define KERNEL(func) func<<< blocks, threads>>>

	BiCGSTAB::BiCGSTAB() {};
	BiCGSTAB::BiCGSTAB(unsigned int N_, double* x, double* x0, double* b,
		SparseMatrixData* A, CudaLaunchSetup kernel_setting, unsigned int reduction_threads)
	{
		N = N_;
		Nbytes = N * sizeof(double);
		threads = kernel_setting.block1d.x;
		blocks = kernel_setting.grid1d.x;

		#define alloc_(ptr) cudaMalloc((void**)&##ptr, Nbytes);  cudaMemset(ptr, 0, Nbytes); 
		alloc_(r); alloc_(r_hat); alloc_(p); alloc_(t); alloc_(s); alloc_(v);

		CR = new CudaReductionM(N, reduction_threads);
		make_graph(x, x0, b, A);
	}
	void BiCGSTAB::solve_directly(double* x, double* x0, double* b, SparseMatrixData* A)
	{
		double rs_host = 1;  k = 0;
		cudaMemset(x, 0, Nbytes);

		// r = b - Ax
		KERNEL(vector_minus_matrix_dot_vector)(r, b, A, x, N);
		// r_hat = r
		KERNEL(vector_set_to_vector)(r_hat, r, N);
		// p = r
		KERNEL(vector_set_to_vector)(p, r, N);

		// rs = r_hat * r
		CR->reduce(r_hat, r, true, ExtraAction::compute_rs_old);

		auto single_iteration = [&]()
		{

			// rs_new = r_hat * r; 		// beta =  (rs_new / rs_old) * (alpha / omega)		// rs_old = rs_new
			CR->reduce(r_hat, r, false, ExtraAction::compute_rs_new_and_beta);

			// p = r + beta * ( p - omega * v)
			KERNEL(vector_add_2vectors)(p, r, p, v, N, KernelCoefficient::beta_and_omega);

			// v = Ap
			KERNEL(matrix_dot_vector)(v, A, p, N);

			// alpha = rs_new / (r_hat * v)
			CR->reduce(r_hat, v, false, ExtraAction::compute_alpha);

			// s = r - alpha * v
			KERNEL(vector_minus_vector)(s, r, v, N, KernelCoefficient::alpha);

			// t = A * s
			KERNEL(matrix_dot_vector)(t, A, s, N);

			// omega = (t * s) / (t * t)

			CR->reduce(t, s, false, ExtraAction::compute_buffer);
			CR->reduce(t, t, false, ExtraAction::compute_omega);

			// x = x + alpha * p + omega * s
			KERNEL(vector_add_2vectors)(x, x, p, s, N, KernelCoefficient::alpha_and_omega);

			// r = s - omega * t
			KERNEL(vector_minus_vector)(r, s, t, N, KernelCoefficient::omega);
		};


		while (true)
		{
			k++;	if (k > 1000000) break;

			single_iteration();

			// check exit by r^2
			if (k < 20 || k % 50 == 0)
			{
				rs_host = CR->reduce(r, r, true, ExtraAction::NONE);
				//if (k > 100000) break;
				if (abs(rs_host) < eps) break;
			}

			//if (k == 20000) break;
			if (k % 1000 == 0) std::cout << k << " " << abs(rs_host) << std::endl;
		}

		//std::cout << k << " " << abs(rs_host) << std::endl;
	}
	void BiCGSTAB::make_graph(double* x, double* x0, double* b, SparseMatrixData* A)
	{
		KernelCoefficient action;

		// 1. rs_new = r_hat * r; 		// beta =  (rs_new / rs_old) * (alpha / omega)		// rs_old = rs_new
		graph.add_graph_as_node(CR->make_graph(r_hat, r, false, ExtraAction::compute_rs_new_and_beta));

		// 2. p = r + beta * ( p - omega * v)
		{
			action = KernelCoefficient::beta_and_omega;
			void* args[] = { &p, &r, &p, &v, &N, &action };
			graph.add_kernel_node(threads, blocks, vector_add_2vectors, args);
		}

		// 3. v = Ap
		{
			void* args[] = { &v, &A, &p, &N };
			graph.add_kernel_node(threads, blocks, matrix_dot_vector, args);
		}

		// 4. alpha = rs_new / (r_hat * v)
		graph.add_graph_as_node(CR->make_graph(r_hat, v, false, ExtraAction::compute_alpha));

		// 5. s = r - alpha * v
		{
			action = KernelCoefficient::alpha;
			void* args[] = { &s, &r, &v, &N, &action };
			graph.add_kernel_node(threads, blocks, vector_minus_vector, args);
		}

		// 6. t = A * s
		{
			void* args[] = { &t, &A, &s, &N };
			graph.add_kernel_node(threads, blocks, matrix_dot_vector, args);
		}

		// 7. omega = (t * s) / (t * t)
		graph.add_graph_as_node(CR->make_graph(t, s, false, ExtraAction::compute_buffer));
		graph.add_graph_as_node(CR->make_graph(t, t, false, ExtraAction::compute_omega));

		// 8. x = x + alpha * p + omega * s
		{
			action = KernelCoefficient::alpha_and_omega;
			void* args[] = { &x, &x, &p, &s, &N, &action };
			graph.add_kernel_node(threads, blocks, vector_add_2vectors, args);
		}

		// 9. r = s - omega * t
		{
			action = KernelCoefficient::omega;
			void* args[] = { &r, &s, &t, &N, &action };
			graph.add_kernel_node(threads, blocks, vector_minus_vector, args);
		}

		graph.instantiate();
	}
	void BiCGSTAB::solve_with_graph(double* x, double* x0, double* b, SparseMatrixData* A)
	{
		double rs_host = 1;  k = 0;
		cudaMemset(x, 0, Nbytes);

		// r = b - Ax
		KERNEL(vector_minus_matrix_dot_vector)(r, b, A, x, N);
		// r_hat = r
		KERNEL(vector_set_to_vector)(r_hat, r, N);
		// p = r
		KERNEL(vector_set_to_vector)(p, r, N);
		// rs = r_hat * r
		CR->reduce(r_hat, r, false, ExtraAction::compute_rs_old);

		while (true)
		{
			k++;	if (k > 1000000) break;
			graph.launch();

			// check exit by r^2
			if (k < 20 || k % 50 == 0)
			{
				rs_host = CR->reduce(r, r, true, ExtraAction::NONE);
				//if (k > 100000) break;
				if (abs(rs_host) < eps) break;
			}

			//if (k == 20000) break;
			if (k % 1000 == 0) std::cout << k << " " << abs(rs_host) << std::endl;
		}
		std::cout << "bicgstab: k = " << k << ", res = " << abs(rs_host) << std::endl;
	}
}