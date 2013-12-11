#include "gpu_optimizer.h"

#include <iostream>

#include <thread>
#include <atomic>
#include <chrono>
#include <system_error>

#include "gpu_problem.h"
#include "gpu_analysis.h"
#include "gpu_optdata.h"
#include "gpu_optimize.h"

// #define FIXED_BATCH_SIZE 48*1024

struct worker {
	std::thread thread;
	std::atomic<bool> cancel;
	std::atomic<uint32_t> iteration;
	worker() : cancel(false) {
	}
};

gpu_optimizer::gpu_optimizer() {
	w = new worker();
}

gpu_optimizer::~gpu_optimizer() {
	delete w;
}

bool gpu_optimizer::is_available() const {
	return gpu_check_requirements();
}

uint32_t gpu_optimizer::get_iteration() const {
	return w->iteration;
}

void gpu_optimizer::optimize(const full_model & fm) {
	// set threshold
	const float delta_threshold = 1e-4f;
	// get information
	const problem & p = fm.get_problem();
	// allocate and initialize gpu memory
	gpu_problem_allocator gp;
	gpu_analysis_allocator ga;
	gpu_optdata_allocator go;
	if (!gp.init(p)) {
		throw std::runtime_error("Failure when trying to allocate and initialize gpu_problem");
	}
	if (!ga.init(p, fm)) {
		throw std::runtime_error("Failure when trying to allocate and initialize gpu_analysis");
	}
    #if defined FIXED_BATCH_SIZE
    uint64_t state_batch_size = FIXED_BATCH_SIZE;
    uint64_t required_byte = go.estimate_memory(p, fm, state_batch_size);
    std::cout << "Using fixed batch size: " << state_batch_size << " ";
    std::cout << "(" << (required_byte / (1024*1024)) << " MB required)" << std::endl;
    #else
	uint64_t state_batch_size = fm.get_system_states();
	size_t free_byte, total_byte;
	cudaError_t err = cudaMemGetInfo( &free_byte, &total_byte );
	if (err == cudaSuccess) {
		uint64_t required_byte = go.estimate_memory(p, fm, state_batch_size);
		// take into account a 10% overhead
		while ((required_byte * 11 / 10) > free_byte)  {
			std::cout << "Batch size of " << state_batch_size << " requires ";
			std::cout << (required_byte / (1024*1024)) << " MB, ";
			std::cout << "but only " << (free_byte / (1024*1024)) << " MB are available" << std::endl;
			state_batch_size /= 2;
			required_byte = go.estimate_memory(p, fm, state_batch_size);
		}
		if (state_batch_size < fm.get_system_states()) {
			std::cout << "Using batched algorithm with batch size " << state_batch_size;
		} else {
			std::cout << "Using non-batched algorithm";
		}
		std::cout << " (" << (required_byte / (1024*1024)) << " MB required)" << std::endl;
	} else {
		throw std::runtime_error("Unable to query available GPU memory");
	}
    #endif
	if (!go.init(p, fm, state_batch_size)) {
		throw std::runtime_error("Failure when trying to allocate and initialize gpu_optdata");
	}
	// start timing
	std::chrono::high_resolution_clock::time_point start, end;
	start = std::chrono::high_resolution_clock::now();
	// initialize optimization
	gpu_optimize_init(*gp.get(), *ga.get(), *go.get());
	w->iteration = 0;
	float delta = 1.0f;
	float cost = 0.0f;
	//  iterations
	while (delta > delta_threshold && !w->cancel) {
		w->iteration += 1;
		gpu_optimize_iteration(
			ignore_revenue, always_allow_reject, check_strict_convergence,
			w->iteration, cost, delta);
		std::cout << "Iteration " << w->iteration << ": delta = " << delta << ", objective = " << cost << std::endl;
	}
	// save solution
	objective = cost;
	gpu_get_policy(policy);
	// end timing
	end = std::chrono::high_resolution_clock::now();
	long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
	// report information
	if (delta < delta_threshold) {
		std::cout << "Value Iteration Algorithm completed in " << w->iteration << " iterations" << std::endl;
	} else {
		std::cout << "Value Iteration Algorithm canceled after " << w->iteration << " iterations" << std::endl;
	}
	std::cout << "Value Iteration Algorithm run for " << ms << " ms" << std::endl;
	std::cout << "Value Iteration Algorithm: average cost: " << objective << std::endl;
}

void gpu_optimizer::start_optimize(const full_model & fm) {
	w->iteration = 0;
	w->cancel = false;
	w->thread = std::thread(&gpu_optimizer::optimize, this, std::cref(fm));
}

void gpu_optimizer::cancel_optimize() {
	w->cancel = true;
}

void gpu_optimizer::join_optimize() {
	try {
		w->thread.join();
	} catch (std::system_error & e) {
		std::cout << e.what() << std::endl;
	}
}
