#ifndef ANVIK_GPU_PROBLEM_H
#define ANVIK_GPU_PROBLEM_H

#include <stdint.h>
#include <cuda_runtime.h>
#include <vector>

struct problem;

struct gpu_problem {
	uint32_t k;
	uint32_t m;
	uint32_t r;
	uint32_t * n;
	uint32_t * capacity;
	uint32_t * requirement;
	float * lambda;
	float * mu;
	float * cost;
	float * revenue;
	__device__ uint32_t get_capacity(uint32_t i, uint32_t h) const {
		return capacity[i * r + h];
	}
	__device__ uint32_t get_requirement(uint32_t j, uint32_t h) const {
		return requirement[j * r + h];
	}
};

class gpu_problem_allocator {
public:
	gpu_problem_allocator();
	~gpu_problem_allocator();
	gpu_problem * init(const problem & p);
	const gpu_problem * get() const {
		return gp;
	}
	gpu_problem * get() {
		return gp;
	}
	void destroy();
private:
	gpu_problem * gp;
};

#endif
