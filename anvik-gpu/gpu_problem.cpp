#include "gpu_problem.h"

#include "problem.h"
#include <iostream>

#define CUDA_CHECK(err) if (err != cudaSuccess) { std::cerr << "CUDA Error at line " << __LINE__ << std::endl; goto error; }

gpu_problem_allocator::gpu_problem_allocator() {
	gp = 0;
}

gpu_problem_allocator::~gpu_problem_allocator() {
	if (gp != 0) {
		destroy();
	}
}

gpu_problem * gpu_problem_allocator::init(const problem & p) {

	gp = new gpu_problem();

	gp->k = p.k;
	gp->m = p.m;
	gp->r = p.r;
	// allocate gpu memory
	CUDA_CHECK( cudaMalloc(&gp->n, gp->k * sizeof(uint32_t)) );
	CUDA_CHECK( cudaMalloc(&gp->capacity, gp->k * gp->r * sizeof(uint32_t)) );
	CUDA_CHECK( cudaMalloc(&gp->requirement, gp->m * gp->r * sizeof(uint32_t)) );
	CUDA_CHECK( cudaMalloc(&gp->lambda, gp->m * sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&gp->mu, gp->m * sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&gp->cost, gp->k * sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&gp->revenue, gp->m * sizeof(float)) );
	// copy data into gpu memory
	CUDA_CHECK( cudaMemcpy(gp->n, &p.n[0], gp->k * sizeof(uint32_t), cudaMemcpyHostToDevice) );
	for (uint32_t i = 0; i < gp->k; ++i) {
		CUDA_CHECK( cudaMemcpy(gp->capacity + gp->r * i, &p.c[i][0], gp->r * sizeof(uint32_t), cudaMemcpyHostToDevice) );
	}
	for (uint32_t j = 0; j < gp->m; ++j) {
		CUDA_CHECK( cudaMemcpy(gp->requirement + gp->r * j, &p.l[j][0], gp->r * sizeof(uint32_t), cudaMemcpyHostToDevice) );
	}
	CUDA_CHECK( cudaMemcpy(gp->lambda, &p.lambda[0], gp->m * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(gp->mu, &p.mu[0], gp->m * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(gp->cost, &p.cost[0], gp->k * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(gp->revenue, &p.revenue[0], gp->m * sizeof(float), cudaMemcpyHostToDevice) );

	goto complete;

error:
	std::cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
	destroy();

complete:
	return gp;
}

void gpu_problem_allocator::destroy() {
	if (gp != 0) {
		// free gpu memory
		cudaFree(gp->n);
		cudaFree(gp->capacity);
		cudaFree(gp->requirement);
		cudaFree(gp->lambda);
		cudaFree(gp->mu);
		cudaFree(gp->cost);
		cudaFree(gp->revenue);
		// free object
		delete gp;
		gp = 0;
	}
}
