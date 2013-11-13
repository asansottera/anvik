#ifndef ANVIK_GPU_ANALYSIS_H
#define ANVIK_GPU_ANALYSIS_H

#include <stdint.h>
#include <cuda_runtime.h>
#include <vector>

struct problem;
class full_model;

struct gpu_analysis {

	/* Number of server groups */
	uint32_t k;
	/* Number of VM classes */
	uint32_t m;
	/* Number of resources */
	uint32_t r;
    /* Total arrival rate */
    float total_arrival_rate;
	/* Number of system states */
	uint64_t system_state_count;
	/* The i-th element is the number of states of the i-th group of servers */
	uint64_t * group_state_count;
	/* The i-th element is the number of states for a single server of the i-th group */
	uint32_t * server_state_count;
	/* The maximum number of states for a server. */
	uint32_t max_server_state_count;
	/* The number of actions. */
	uint32_t action_count;
	/* The i-th element is a column-major matrix of size m x server_state_count[i].
	   The (j,w) element is the number of VMs of class j allocated on a server of class i in state w. */
	uint32_t ** server_states;
	/* The i-th element is a column-major matrix of size m x server_state_count[i].
	   The (j,w) element is the next state for a server of class i in state w when a new VM of class j is allocated onto i.
	   If the next state is equal to w, there is no room for the VM and the transition is infeasible. */
	uint32_t ** arrival_transitions;
	/* The i-th element is a column-major matrix of size m x server_state_count[i].
	   The (j,w) element is the next state for a server of class i in state w when a new VM of class j is deallocated from i.
	   If the next state is equal to w, there is such VM and the transition is infeasible. */
	uint32_t ** departure_transitions;
	/* A column-major matrix of size m x action_count.
	   The (j,a) element is true if VMs of class j are dropped when taking action a. */
	bool * action_drop;
	/* A column-major matrix of size m x action_count.
	   The (j,a) element is the server group on which a VM class j is allocated when taking action a.
	   The value is meaningful only if action_drop(j,a) is false. */
	uint32_t * action_server_group;
	/* A column-major matrix of size m x action_count.
	   The (j,a) element is the state of the server on which a VM class j is allocated when taking action a.
	   The value is meaningful only if action_drop(j,a) is false. */
	uint32_t * action_server_state;
	/* The i-th pointers is a (n+1) x (w+1) matrix with the multichoose coefficients relevant to the i-th group of servers. */
	uint64_t ** multichoose_tables;

	/* Get the number of server states for servers of group i. */
	__device__ uint32_t get_server_states(uint32_t i) const {
		return server_state_count[i];
	}
	/* Get a vector whose j-th element is the number of VMs of class j allocated on a server of group i in state w. */
	__device__ uint32_t * get_server_state(uint32_t i, uint32_t w) const {
		return server_states[i] + w * m;
	}
	/* Get a vector whose j-th element is the next state of a server of group i, currently in state w,
	   when a VM of class j is allocated on it. */
	__device__ uint32_t * get_arrival_transitions(uint32_t i, uint32_t w) const {
		return arrival_transitions[i] + w * m;
	}
	/* Get a vector whose j-th element is the next state of a server of group i, currently in state w,
	   when a VM of class j is deallocated from it. */
	__device__ uint32_t * get_departure_transitions(uint32_t i, uint32_t w) const {
		return departure_transitions[i] + w * m;
	}
	/* Get a vector whose j-th element is true if action a drops VMs of class j. */
	__device__ bool * get_action_drop(uint32_t a) const {
		return action_drop + a * m;
	}
	/* Get a vector whose j-th element is the server group on which action a allocates VMs of class j. */
	__device__ uint32_t * get_action_server_group(uint32_t a) const {
		return action_server_group + a * m;
	}
	/* Get a vector whose j-th element is the server state on which action a allocates VMs of class j. */
	__device__ uint32_t * get_action_server_state(uint32_t a) const {
		return action_server_state + a * m;
	}
};

class gpu_analysis_allocator {
public:
	gpu_analysis_allocator();
	~gpu_analysis_allocator();
	gpu_analysis * init(const problem & p, const full_model & fm);
	const gpu_analysis * get() const {
		return ga;
	}
    gpu_analysis * get() {
		return ga;
	}
	void destroy();
private:
	gpu_analysis * ga;
	std::vector<uint32_t *> server_states_ptrs;
	std::vector<uint32_t *> arrival_transitions_ptrs;
	std::vector<uint32_t *> departure_transitions_ptrs;
	std::vector<uint64_t *> multichoose_tables_ptrs;
};

#endif
