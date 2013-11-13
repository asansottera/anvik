#ifndef ANVIK_GPU_OPTDATA_H
#define ANVIK_GPU_OPTDATA_H

#include <stdint.h>
#include <cuda_runtime.h>
#include <vector>

struct problem;
class full_model;

struct gpu_optdata {
	uint64_t state_batch_size;
	uint64_t * current_system_state;
	uint32_t ** current_group_states;
	bool * accept_vm;
	uint32_t * running_vm;
	float * cost;
	float * revenue;
	float * departure_rate;
	uint64_t ** arrival_transitions;
	uint64_t ** arrival_transitions_pointers;
	uint64_t ** departure_transitions;
	uint64_t ** departure_transitions_pointers;
	float * via_w_1;
	float * via_w_2;
	uint32_t * via_policy;
};

class gpu_optdata_allocator {
public:
	gpu_optdata_allocator();
	~gpu_optdata_allocator();
	/* Returns an estimate of the required memory in bytes. */
	uint64_t estimate_memory(const problem & p, const full_model & fm, uint64_t stateBatchSize);
	/* Releases memory. */
	void destroy();
	/* Allocates memory. */
	gpu_optdata * init(const problem & p, const full_model & fm, uint64_t stateBatchSize);
	/* Gets the data struture. */
	const gpu_optdata * get() const {
		return go;
	}
	/* Gets the data struture. */
	gpu_optdata * get() {
		return go;
	}
private:
	gpu_optdata * go;
	std::vector<uint32_t *> current_group_states_ptrs;
	std::vector<uint64_t *> arrival_transitions_ptrs;
	std::vector<uint64_t *> departure_transitions_ptrs;
};

#endif
