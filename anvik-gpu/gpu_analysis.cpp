#include "gpu_analysis.h"

#include "problem.h"
#include "full_model.h"
#include <vector>
#include <iostream>

#define CUDA_CHECK(err) if (err != cudaSuccess) { std::cerr << "CUDA Error at line " << __LINE__ << std::endl; goto error; }

gpu_analysis_allocator::gpu_analysis_allocator() {
	ga = 0;
}

gpu_analysis_allocator::~gpu_analysis_allocator() {
	if (ga != 0) {
		destroy();
	}
}

gpu_analysis * gpu_analysis_allocator::init(const problem & p, const full_model & fm) {

	ga = new gpu_analysis();

    // constants
	ga->k = p.k;
	ga->m = p.m;
	ga->r = p.r;
	ga->system_state_count = fm.get_system_states();
	ga->action_count = static_cast<uint32_t>(fm.get_actions());
    ga->total_arrival_rate = 0.0f;
    for (uint32_t j = 0; j < p.m; ++j) {
        ga->total_arrival_rate += p.lambda[j];
    }
	ga->max_server_state_count = 0;
	for (uint32_t i = 0; i < p.k; ++i) {
		ga->max_server_state_count = std::max<uint32_t>(ga->max_server_state_count, fm.get_server_states(i));
	}

	// layout information nicely in host memory
	// this allows transfering data with the least amount of cudaMemcpy
	std::vector<uint64_t> v_group_state_count(p.k);
	std::vector<uint32_t> v_server_state_count(p.k);
	for (uint32_t i = 0; i < p.k; ++i) {
		v_group_state_count[i] = fm.get_group_states(i);
		v_server_state_count[i] = static_cast<uint32_t>(fm.get_server_states(i));
	}
	std::vector<uint8_t> v_action_drop(ga->m * ga->action_count);
	std::vector<uint32_t> v_action_server_group(ga->m * ga->action_count);
	std::vector<uint32_t> v_action_server_state(ga->m * ga->action_count);
	for (uint32_t a = 0; a < ga->action_count; ++a) {
		const std::vector<full_model::action> & actionv = fm.get_action(a);
		for (uint32_t j = 0; j < ga->m; ++j) {
			v_action_drop[a * ga->m + j] = actionv[j].drop;
			v_action_server_group[a * ga->m + j] = actionv[j].server_type;
			v_action_server_state[a * ga->m + j] = actionv[j].server_alloc;
		}
	}
	std::vector<matrix<uint64_t>> v_multichoose_tables(p.k);
	for (uint32_t i = 0; i < p.k; ++i) {
		uint32_t N = p.n[i];
		uint32_t W = static_cast<uint32_t>(fm.get_server_states(i));
		multichoose::compute(N, W, v_multichoose_tables[i]);
	}
	// prepare host memory to hold gpu pointers
	server_states_ptrs.resize(ga->k);
	arrival_transitions_ptrs.resize(ga->k);
	departure_transitions_ptrs.resize(ga->k);
	multichoose_tables_ptrs.resize(ga->k);

	// copy the number of server states and group states for each
	CUDA_CHECK( cudaMalloc(&ga->group_state_count, p.k * sizeof(uint64_t)) );
	CUDA_CHECK( cudaMalloc(&ga->server_state_count, p.k * sizeof(uint32_t)) );
	CUDA_CHECK( cudaMemcpy(ga->group_state_count, &v_group_state_count[0], p.k * sizeof(uint64_t), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(ga->server_state_count, &v_server_state_count[0], p.k * sizeof(uint32_t), cudaMemcpyHostToDevice) );

	// copy server state and transition information
	for (uint32_t i = 0; i < p.k; ++i) {
		// we have count * m matrix for each server group
		uint32_t count = static_cast<uint32_t>(fm.get_server_states(i));
		// copy memory for server states (feasible allocations)
		CUDA_CHECK( cudaMalloc(&server_states_ptrs[i], count * ga->m * sizeof(uint32_t)) );
		for (uint32_t w = 0; w < count; ++w) {
			const std::vector<uint32_t> & server_state = fm.get_server_state(i, w);
			CUDA_CHECK( cudaMemcpy(server_states_ptrs[i] + w * ga->m, &server_state[0], ga->m * sizeof(uint32_t), cudaMemcpyHostToDevice) );
		}
		// copy memory for arrival transitions
		CUDA_CHECK( cudaMalloc(&arrival_transitions_ptrs[i], count * ga->m * sizeof(uint32_t)) );
		for (uint32_t w = 0; w < count; ++w) {
			const std::vector<uint32_t> & arrtran = fm.get_arrival_transitions(i, w);
			CUDA_CHECK( cudaMemcpy(arrival_transitions_ptrs[i] + w * ga->m, &arrtran[0], ga->m * sizeof(uint32_t), cudaMemcpyHostToDevice) );
		}
		// copy memory for departure transitions
		CUDA_CHECK( cudaMalloc(&departure_transitions_ptrs[i], count * ga->m * sizeof(uint32_t)) );
		for (uint32_t w = 0; w < count; ++w) {
			const std::vector<uint32_t> & deptran = fm.get_departure_transitions(i, w);
			CUDA_CHECK( cudaMemcpy(departure_transitions_ptrs[i] + w * ga->m, &deptran[0], ga->m * sizeof(uint32_t), cudaMemcpyHostToDevice) );
		}
		// copy memory for multichoose tables
		CUDA_CHECK( cudaMalloc(&multichoose_tables_ptrs[i], v_multichoose_tables[i].size() * sizeof(uint64_t)) );
		CUDA_CHECK( cudaMemcpy(multichoose_tables_ptrs[i], v_multichoose_tables[i].raw(), v_multichoose_tables[i].size() * sizeof(uint64_t), cudaMemcpyHostToDevice) );
	}
	// copy device pointers to the GPU
	CUDA_CHECK( cudaMalloc(&ga->server_states, ga->k * sizeof(uint32_t *)) );
	CUDA_CHECK( cudaMalloc(&ga->arrival_transitions, ga->k * sizeof(uint32_t *)) );
	CUDA_CHECK( cudaMalloc(&ga->departure_transitions, ga->k * sizeof(uint32_t *)) );
	CUDA_CHECK( cudaMalloc(&ga->multichoose_tables, ga->k * sizeof(uint64_t *)) );
	CUDA_CHECK( cudaMemcpy(ga->server_states, &server_states_ptrs[0], ga->k * sizeof(uint32_t *), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(ga->arrival_transitions, &arrival_transitions_ptrs[0], ga->k * sizeof(uint32_t *), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(ga->departure_transitions, &departure_transitions_ptrs[0], ga->k * sizeof(uint32_t *), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(ga->multichoose_tables, &multichoose_tables_ptrs[0], ga->k * sizeof(uint64_t *), cudaMemcpyHostToDevice) );
	// copy action information
	CUDA_CHECK( cudaMalloc(&ga->action_drop, ga->m * ga->action_count * sizeof(bool)) );
	CUDA_CHECK( cudaMalloc(&ga->action_server_group, ga->m * ga->action_count * sizeof(uint32_t)) );
	CUDA_CHECK( cudaMalloc(&ga->action_server_state, ga->m * ga->action_count * sizeof(uint32_t)) );
	CUDA_CHECK( cudaMemcpy(ga->action_drop, &v_action_drop[0], ga->m * ga->action_count * sizeof(bool), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(ga->action_server_group, &v_action_server_group[0], ga->m * ga->action_count * sizeof(uint32_t), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(ga->action_server_state, &v_action_server_state[0], ga->m * ga->action_count * sizeof(uint32_t), cudaMemcpyHostToDevice) );

	goto complete;

error:
	std::cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
	destroy();

complete:
	return ga;
}

void gpu_analysis_allocator::destroy() {
	if (ga != 0) {
		// free device memory
		cudaFree(ga->server_state_count);
		cudaFree(ga->group_state_count);
		for (uint32_t i = 0; i < ga->k; ++i) {
			cudaFree(server_states_ptrs[i]);
			cudaFree(arrival_transitions_ptrs[i]);
			cudaFree(departure_transitions_ptrs[i]);
			cudaFree(multichoose_tables_ptrs[i]);
		}
		cudaFree(ga->server_states);
		cudaFree(ga->arrival_transitions);
		cudaFree(ga->departure_transitions);
		cudaFree(ga->multichoose_tables);
		cudaFree(ga->action_drop);
		cudaFree(ga->action_server_group);
		cudaFree(ga->action_server_state);
		// clear pointers
		ga->server_state_count = 0;
		ga->group_state_count = 0;
		server_states_ptrs.clear();
		arrival_transitions_ptrs.clear();
		departure_transitions_ptrs.clear();
		multichoose_tables_ptrs.clear();
		ga->server_states = 0;
		ga->arrival_transitions = 0;
		ga->departure_transitions = 0;
		ga->multichoose_tables = 0;
		ga->action_drop = 0;
		ga->action_server_group = 0;
		ga->action_server_state = 0;
		// delete gpu_analysis
		delete ga;
		ga = 0;
	}
}
