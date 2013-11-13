#include "gpu_optdata.h"

#include "problem.h"
#include "full_model.h"
#include <iostream>
#include <cstdint>

#define CUDA_CHECK(err) if (err != cudaSuccess) { std::cerr << "CUDA Error at line " << __LINE__ << std::endl; goto error; }

gpu_optdata_allocator::gpu_optdata_allocator() {
	go = 0;
}

gpu_optdata_allocator::~gpu_optdata_allocator() {
	if (go != 0) {
		destroy();
	}
}

uint64_t gpu_optdata_allocator::estimate_memory(const problem & p, const full_model & fm, uint64_t stateBatchSize) {
	uint64_t bytes = 0;
	bytes += stateBatchSize * p.k * sizeof(uint64_t);
	for (uint32_t i = 0; i < p.k; ++i) {
		bytes += stateBatchSize * fm.get_server_states(i) * sizeof(uint32_t);
	}
	bytes += p.k * sizeof(uint32_t *);
	bytes += stateBatchSize * p.m * sizeof(bool);
	bytes += stateBatchSize * p.m * sizeof(uint32_t);
	bytes += stateBatchSize * sizeof(float);
	bytes += stateBatchSize * sizeof(float);
	bytes += stateBatchSize * sizeof(float);
	bytes += fm.get_system_states() * sizeof(float);
	bytes += fm.get_system_states() * sizeof(float);
	bytes += fm.get_system_states() * sizeof(uint32_t);
	for (uint32_t i = 0; i < p.k; ++i) {
		bytes += stateBatchSize * fm.get_server_states(i) * p.m * sizeof(uint64_t);
		bytes += stateBatchSize * fm.get_server_states(i) * p.m * sizeof(uint64_t);
	}
	bytes += p.k * sizeof(uint64_t *);
	bytes += p.k * sizeof(uint64_t *);
	bytes += fm.get_actions() * p.m * sizeof(uint64_t *);
	bytes += fm.get_actions() * p.m * sizeof(uint64_t *);
	return bytes;
}

gpu_optdata * gpu_optdata_allocator::init(const problem & p, const full_model & fm, uint64_t stateBatchSize) {
	
	go = new gpu_optdata();

	go->state_batch_size = stateBatchSize;

	CUDA_CHECK( cudaMalloc(&go->current_system_state, stateBatchSize * p.k * sizeof(uint64_t)) );
	current_group_states_ptrs.resize(p.k);
	for (uint32_t i = 0; i < p.k; ++i) {
		CUDA_CHECK( cudaMalloc(&current_group_states_ptrs[i], stateBatchSize * fm.get_server_states(i) * sizeof(uint32_t)) );
	}
	CUDA_CHECK( cudaMalloc(&go->current_group_states, p.k * sizeof(uint32_t *)) );
	CUDA_CHECK( cudaMemcpy(go->current_group_states, &current_group_states_ptrs[0], p.k * sizeof(uint32_t *), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMalloc(&go->accept_vm, stateBatchSize * p.m * sizeof(bool)) );
	CUDA_CHECK( cudaMalloc(&go->running_vm, stateBatchSize * p.m * sizeof(uint32_t)) );
	CUDA_CHECK( cudaMalloc(&go->cost, stateBatchSize * sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&go->revenue, stateBatchSize * sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&go->departure_rate, stateBatchSize * sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&go->via_w_1, fm.get_system_states() * sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&go->via_w_2, fm.get_system_states() * sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&go->via_policy, fm.get_system_states() * sizeof(uint32_t)) );
	arrival_transitions_ptrs.resize(p.k);
	departure_transitions_ptrs.resize(p.k);
	for (uint32_t i = 0; i < p.k; ++i) {
		CUDA_CHECK( cudaMalloc(&arrival_transitions_ptrs[i], stateBatchSize * fm.get_server_states(i) * p.m * sizeof(uint64_t)) );
		CUDA_CHECK( cudaMalloc(&departure_transitions_ptrs[i], stateBatchSize * fm.get_server_states(i) * p.m * sizeof(uint64_t)) );
	}
	CUDA_CHECK( cudaMalloc(&go->arrival_transitions, p.k * sizeof(uint64_t *)) );
	CUDA_CHECK( cudaMalloc(&go->departure_transitions, p.k * sizeof(uint64_t *)) );
	CUDA_CHECK( cudaMemcpy(go->arrival_transitions, &arrival_transitions_ptrs[0], p.k * sizeof(uint64_t *), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(go->departure_transitions, &departure_transitions_ptrs[0], p.k * sizeof(uint64_t *), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMalloc(&go->arrival_transitions_pointers, fm.get_actions() * p.m * sizeof(uint64_t *)) );
	CUDA_CHECK( cudaMalloc(&go->departure_transitions_pointers, fm.get_actions() * p.m * sizeof(uint64_t *)) );

	goto complete;

error:
	std::cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
	destroy();

complete:
	return go;
}

void gpu_optdata_allocator::destroy() {
	if (go != 0) {
		cudaFree(go->current_system_state);
		for (uint32_t i = 0; i < current_group_states_ptrs.size(); ++i) {
			cudaFree(current_group_states_ptrs[i]);
		}
		current_group_states_ptrs.clear();
		cudaFree(go->current_group_states);
		cudaFree(go->accept_vm);
		cudaFree(go->running_vm);
		cudaFree(go->cost);
		cudaFree(go->revenue);
		cudaFree(go->departure_rate);
		cudaFree(go->via_w_1);
		cudaFree(go->via_w_2);
		cudaFree(go->via_policy);
		for (uint32_t i = 0; i < arrival_transitions_ptrs.size(); ++i) {
			cudaFree(arrival_transitions_ptrs[i]);
			cudaFree(departure_transitions_ptrs[i]);
		}
		arrival_transitions_ptrs.clear();
		departure_transitions_ptrs.clear();
		cudaFree(go->arrival_transitions_pointers);
		cudaFree(go->departure_transitions_pointers);
		delete go;
        go = 0;
	}
}
