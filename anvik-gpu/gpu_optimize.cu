#include "gpu_optimize.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gpu_reduction.h"

#include <cstdio>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cfloat>

#include "gpu_problem.h"
#include "gpu_analysis.h"
#include "gpu_optdata.h"

gpu_problem  cgp;
gpu_analysis cga;
gpu_optdata  cgo;
float        ctau;

__constant__ gpu_problem  gp;
__constant__ gpu_analysis ga;
__constant__ gpu_optdata  go;
__constant__ float        tau;

__device__ void gpu_get_state(uint64_t z, uint32_t idx) {
    uint64_t remainder = z;
    uint64_t factor = 1;
    uint32_t k = gp.k;
    for (uint32_t i = 0; i < k; ++i) {
        factor *= ga.group_state_count[i];
    }
    for (uint32_t i = 0; i < k; ++i) {
        uint32_t group = k - i - 1;
        factor /= ga.group_state_count[group];
		go.current_system_state[go.state_batch_size * group + idx] = remainder / factor;
        remainder = remainder % factor;
    }
}

__device__ void gpu_get_group_state(uint64_t z, uint32_t idx, uint32_t i, uint64_t zi) {
	uint64_t * multichoose_table = ga.multichoose_tables[i];
    // number of servers in the group
    uint32_t n = gp.n[i];
    // number of states for a server in the group
    uint32_t server_states = ga.server_state_count[i];
    // initialize group state
    for (uint32_t w = 0; w < server_states; ++w) {
		go.current_group_states[i][w * go.state_batch_size + idx] = 0;
    }
    go.current_group_states[i][0 * go.state_batch_size + idx] = n;
    // get group state
    uint32_t remaining = 0;
    uint32_t bin = 0;
    uint64_t current = 0;
    while (current != zi) {
        go.current_group_states[i][bin * go.state_batch_size + idx] -= 1;
        remaining += 1;
        uint32_t mc = multichoose_table[(server_states-(bin+1))*(n+1) + remaining];
        uint64_t next = current + mc;
        if (next > zi) {
            bin += 1;
            go.current_group_states[i][bin * go.state_batch_size + idx] = remaining;
            remaining = 0;
            current += 1;
        } else if (next < zi) {
            current = next;
        } else {
            current = next;
            go.current_group_states[i][(server_states-1) * go.state_batch_size + idx] = remaining;
        }
    }
}

__global__ void gpu_init_state_info_kernel(uint64_t z_first) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t z = z_first + idx;
	if (idx < go.state_batch_size && z < ga.system_state_count) {
        uint32_t k = gp.k;
        uint32_t m = gp.m;
        // init system state vector into global memory
        gpu_get_state(z, idx);
        // init cost, revenue and departure rate
        float cost = 0.0f;
        float revenue = 0.0f;
        float departure_rate = 0.0f;
        // for each virtual machine class
        for (uint32_t j = 0; j < m; ++j) {
            // init whether new virtual machines of this class are accepted or not
			go.accept_vm[j * go.state_batch_size + idx] = false;
            // init number of running virtual machines
            go.running_vm[j * go.state_batch_size + idx] = 0;
        }
        // for each server group
        for (uint32_t i = 0; i < k; ++i) {
            uint32_t server_states = ga.server_state_count[i];
			// state of group i
			uint64_t i_state_idx = go.current_system_state[i * go.state_batch_size + idx];
            // init group state vector into global memory
            gpu_get_group_state(z, idx, i, i_state_idx);
            for (uint32_t w = 0; w < server_states; ++w) {
				uint32_t servers = go.current_group_states[i][w * go.state_batch_size + idx];
                // update number of running virtual machines
                for (uint32_t j = 0; j < m; ++j) {
					go.running_vm[j * go.state_batch_size + idx] += servers * ga.get_server_state(i,w)[j];
                    go.accept_vm[j * go.state_batch_size + idx]  |= (servers > 0 && ga.get_arrival_transitions(i,w)[j] != w);
                }
            }
            // update cost
			uint32_t idle_servers = go.current_group_states[i][0 * go.state_batch_size + idx];
            cost += (gp.n[i] - idle_servers) * gp.cost[i];
        }
        // compute revenue and departure rate
        for (uint32_t j = 0; j < m; ++j) {
            uint32_t running = go.running_vm[j * go.state_batch_size + idx];
            revenue += running * gp.revenue[j];
            departure_rate += running * gp.mu[j];
        }
        // write cost, revenue and departure rate in global memory
        go.cost[idx] = cost;
        go.revenue[idx] = revenue;
        go.departure_rate[idx] = departure_rate;
	}
}

__device__ uint64_t gpu_compute_next_state(uint64_t z, uint32_t i, uint32_t w, uint32_t w_next, uint64_t z_first) {
	uint32_t offset = 0;
	uint32_t server_state_count = ga.server_state_count[i];
	// get current group state index for group i
	uint64_t current_group_state_idx = go.current_system_state[go.state_batch_size * i + (z-z_first)];
	uint32_t * current_group_state_address = go.current_group_states[i] + (z-z_first);
	// alternative way to compute next group state
	uint64_t * mc = ga.multichoose_tables[i];
	uint32_t rows = gp.n[i] + 1;
	uint64_t next_group_state_idx = current_group_state_idx;
	if (w < w_next) {
		uint32_t remaining = gp.n[i];
		offset = 0;
		for (uint32_t w2 = 0; w2 < w; ++w2) {
			remaining -= current_group_state_address[offset];
			offset += go.state_batch_size;
		}
		for (uint32_t w2 = w; w2 < w_next; ++w2) {
			remaining -= current_group_state_address[offset];
			uint32_t col = server_state_count-w2-1;
			uint32_t row = remaining;
			next_group_state_idx += mc[col * rows + row];
			offset += go.state_batch_size;
		}
	} else if (w > w_next) {
		uint32_t remaining = gp.n[i];
		offset = 0;
		for (uint32_t w2 = 0; w2 < w_next; ++w2) {
			remaining -= current_group_state_address[offset];
			offset += go.state_batch_size;
		}
		for (uint32_t w2 = w_next; w2 < w; ++w2) {
			remaining -= current_group_state_address[offset];
			uint32_t col = server_state_count-w2-1;
			uint32_t row = remaining - 1;
			next_group_state_idx -= mc[col * rows + row];
			offset += go.state_batch_size;
		}
	}
	// compute next system state
	uint64_t factor = 1;
	for (uint32_t i2 = 0; i2 < i; ++i2) {
		factor *= ga.group_state_count[i2];
	}
	uint64_t z_next = z + (next_group_state_idx - current_group_state_idx) * factor;
	return z_next;
}

__device__ void gpu_compute_departure_cost(uint64_t z, unsigned iteration, uint64_t z_first) {
	// get values needed to reconstruct u
	float * via_w_old;
	if (iteration % 2 == 0) {
		via_w_old = go.via_w_2;
	} else {
		via_w_old = go.via_w_1;
	}
	float via_w_old_ref = via_w_old[0];
	// compute partial w
	float via_w_z = 0.0f;
	for (uint32_t i = 0; i < gp.k; ++i) {
		uint64_t * dept = go.departure_transitions[i];
		uint32_t * group_state = go.current_group_states[i];
		uint32_t server_state_count = ga.server_state_count[i];
		uint64_t woffset = 0;
		for (uint32_t w = 0; w < server_state_count; ++w) {
			uint32_t servers = group_state[woffset + (z - z_first)];
			if (servers > 0) {
				uint32_t * server_state = ga.get_server_state(i, w);
				uint64_t joffset = 0;
				for (uint32_t j = 0; j < gp.m; ++j) {
					uint32_t vms = server_state[j];
					if (vms > 0) {
						float factor = servers * vms * tau * gp.mu[j];
						// compute index of next system state
						uint64_t z_next = dept[joffset + (z - z_first)];
						// update cost
						float via_u_z_next = via_w_old[z_next] - via_w_old_ref;
						via_w_z += factor * via_u_z_next;
					}
					joffset += go.state_batch_size;
				}
			}
			woffset += go.state_batch_size;
			dept += gp.m * go.state_batch_size;
		}
	}
	if (iteration % 2 == 0) {
		go.via_w_1[z] += via_w_z;
	} else {
		go.via_w_2[z] += via_w_z;
	}
}

__device__ void gpu_compute_departure_cost_unroll2(uint64_t z1, uint64_t z2, unsigned iteration, uint64_t z_first) {
	// get values needed to reconstruct u
	float * via_w_old;
	if (iteration % 2 == 0) {
		via_w_old = go.via_w_2;
	} else {
		via_w_old = go.via_w_1;
	}
	float via_w_old_ref = via_w_old[0];
	// compute partial w
	float via_w_z1 = 0.0f;
	float via_w_z2 = 0.0f;
	for (uint32_t i = 0; i < gp.k; ++i) {
		uint64_t * dept = go.departure_transitions[i];
		uint32_t * group_state = go.current_group_states[i];
		uint32_t server_state_count = ga.server_state_count[i];
		uint64_t woffset = 0;
		for (uint32_t w = 0; w < server_state_count; ++w) {
			uint32_t servers1 = group_state[woffset + (z1 - z_first)];
			uint32_t servers2 = group_state[woffset + (z2 - z_first)];
			uint32_t * server_state = ga.get_server_state(i, w);
			if (servers1 > 0 || servers2 > 0) {
				uint64_t joffset = 0;
				for (uint32_t j = 0; j < gp.m; ++j) {
					uint32_t vms = server_state[j];
					if (vms > 0) {
						float factor = vms * tau * gp.mu[j];
						// compute departure rate
						float factor1 = servers1 * factor;
						float factor2 = servers2 * factor;
						// compute index of next system state
						uint64_t z1_next = dept[joffset + (z1 - z_first)];
						uint64_t z2_next = dept[joffset + (z2 - z_first)];
						// update cost
						float via_u_z1_next = via_w_old[z1_next] - via_w_old_ref;
						float via_u_z2_next = via_w_old[z2_next] - via_w_old_ref;
						via_w_z1 += factor1 * via_u_z1_next;
						via_w_z2 += factor2 * via_u_z2_next;
					}
					joffset += go.state_batch_size;
				}
			}
			woffset += go.state_batch_size;
			dept += gp.m * go.state_batch_size;
		}
	}
	if (iteration % 2 == 0) {
		go.via_w_1[z1] += via_w_z1;
		go.via_w_1[z2] += via_w_z2;
	} else {
		go.via_w_2[z1] += via_w_z1;
		go.via_w_2[z2] += via_w_z2;
	}
}

__global__ void gpu_compute_transitions_kernel(uint64_t z_first) {
	uint64_t z = z_first + (blockDim.x * blockIdx.x + threadIdx.x);
	if ((z-z_first) < go.state_batch_size && z < ga.system_state_count) {
		// compute transitions from this state
		for (uint32_t i = 0; i < gp.k; ++i) {
			const uint32_t server_states = ga.server_state_count[i];
			uint64_t * wjarrivt = go.arrival_transitions[i];
			uint64_t * wjdept = go.departure_transitions[i];
			for (uint32_t w = 0; w < server_states; ++w) {
				const uint32_t servers = go.current_group_states[i][w * go.state_batch_size + (z-z_first)];
				for (uint32_t j = 0; j < gp.m; ++j) {
					if (servers > 0) {
						// compute next server state
						const uint32_t w_next_arrival = ga.get_arrival_transitions(i,w)[j];
						const uint32_t w_next_departure = ga.get_departure_transitions(i,w)[j];
						if (w != w_next_arrival) {
							// compute next system state
							uint64_t z_next_arrival = gpu_compute_next_state(z, i, w, w_next_arrival, z_first);
							// store next state
							wjarrivt[(z-z_first)] = z_next_arrival;
						} else {
							wjarrivt[(z-z_first)] = z;
						}
						if (w != w_next_departure) {
							// compute next system state
							uint64_t z_next_departure = gpu_compute_next_state(z, i, w, w_next_departure, z_first);
							// store next state
							wjdept[(z-z_first)] = z_next_departure;
						} else {
							wjdept[(z-z_first)] = z;
						}
					} else {
						wjarrivt[(z-z_first)] = z;
						wjdept[(z-z_first)] = z;
					}
					wjarrivt += go.state_batch_size;
					wjdept += go.state_batch_size;
				}
			}
		}
	}
}

__global__ void gpu_compute_transition_pointers_kernel() {
	uint64_t a = blockDim.x * blockIdx.x + threadIdx.x;
	if (a < ga.action_count) {
		const bool * drop = ga.get_action_drop(a);
		const uint32_t * server_group = ga.get_action_server_group(a);
		const uint32_t * server_state = ga.get_action_server_state(a);
		for (uint32_t j = 0; j < gp.m; ++j) {
			if (!drop[j]) {
				uint32_t i = server_group[j];
				uint32_t w = server_state[j];
				uint64_t * tptr = go.arrival_transitions[i] + w * gp.m * go.state_batch_size + j * go.state_batch_size;
				go.arrival_transitions_pointers[a * gp.m + j] = tptr;
			}
		}
	}
}

template<bool alwaysAllowReject>
__device__ void gpu_compute_arrival_cost(uint64_t z, unsigned iteration, uint64_t z_first) {
	// get values needed to reconstruct u
	float * via_w_old;
	if (iteration % 2 == 0) {
		via_w_old = go.via_w_2;
	} else {
		via_w_old = go.via_w_1;
	}
	float via_w_old_ref = via_w_old[0];
	// compute best action
	float min_action_cost = FLT_MAX;
	uint32_t best_action = 0;
	for (uint32_t a = 0; a < ga.action_count; ++a) {
		// initialize cost
		float cost = 0.0f;
		// get action information
		const bool * drop = ga.get_action_drop(a);
		// compute feasibility
		bool feasible = true;
		for (uint32_t j = 0; j < gp.m; ++j) {
			bool accept_vm = go.accept_vm[go.state_batch_size * j + (z-z_first)];
			const uint64_t * ptr = go.arrival_transitions_pointers[a * gp.m + j];
			feasible = feasible && ( (accept_vm && !drop[j] && ptr[z-z_first] != z) || (accept_vm && drop[j] && alwaysAllowReject) || (!accept_vm && drop[j]) );
		}
		// compute cost
		if (feasible) {
			for (uint32_t j = 0; j < gp.m; ++j) {
				uint64_t z_next = z;
				if (!drop[j]) {
					const uint64_t * ptr = go.arrival_transitions_pointers[a * gp.m + j];
					z_next = ptr[z-z_first];
				}
				float via_u_z_next = via_w_old[z_next] - via_w_old_ref;
				cost += tau * gp.lambda[j] * via_u_z_next;
			}
			if (cost < min_action_cost) {
				min_action_cost = cost;
				best_action = a;
			}
		}
	}
	// update policy
	go.via_policy[z] = best_action;
	// update cost
	if (iteration % 2 == 0) {
		go.via_w_1[z] += min_action_cost;
	} else {
		go.via_w_2[z] += min_action_cost;
	}
}

template<bool alwaysAllowReject>
__device__ void gpu_compute_arrival_cost_unroll2(uint64_t z1, uint64_t z2, unsigned iteration, uint64_t z_first) {
	// get values needed to reconstruct u
	float * via_w_old;
	if (iteration % 2 == 0) {
		via_w_old = go.via_w_2;
	} else {
		via_w_old = go.via_w_1;
	}
	float via_w_old_ref = via_w_old[0];
	// setup pointers into shared memory
	extern __shared__ bool sh[];
	bool * accept_vm1_sh = &sh[0];
	bool * accept_vm2_sh = &sh[gp.m * blockDim.x];
	bool * drop_sh = &sh[2 * gp.m * blockDim.x];
	// store state information in shared memory: reused for all actions
	for (uint32_t j = 0; j < gp.m; ++j) {
		accept_vm1_sh[j * blockDim.x + threadIdx.x] = go.accept_vm[go.state_batch_size * j + (z1-z_first)];
		accept_vm2_sh[j * blockDim.x + threadIdx.x] = go.accept_vm[go.state_batch_size * j + (z2-z_first)];
	}
	// compute best action
	float min_action_cost1 = FLT_MAX;
	float min_action_cost2 = FLT_MAX;
	uint32_t best_action1 = 0;
	uint32_t best_action2 = 0;
	for (uint32_t a = 0; a < ga.action_count; ++a) {
		// get action information
		const bool * drop = ga.get_action_drop(a);
		// compute feasibility
		bool feasible1 = true;
		bool feasible2 = true;
		for (uint32_t j = 0; j < gp.m; ++j) {
			uint32_t shoffset = j * blockDim.x + threadIdx.x;
			// load from global memory
			bool dropj = drop[j];
			// cache in shared mmeory to reuse in cost calculation
			drop_sh[shoffset] = dropj;
			// load state information from shared memory
			bool accept_vm1 = accept_vm1_sh[shoffset];
			bool accept_vm2 = accept_vm2_sh[shoffset];
			const uint64_t * ptr = go.arrival_transitions_pointers[a * gp.m + j];
			feasible1 = feasible1 && ( (accept_vm1 && !dropj && ptr[z1-z_first] != z1) || (accept_vm1 && dropj && alwaysAllowReject) || (!accept_vm1 && dropj) );
			feasible2 = feasible2 && ( (accept_vm2 && !dropj && ptr[z2-z_first] != z2) || (accept_vm2 && dropj && alwaysAllowReject) || (!accept_vm2 && dropj) );
		}
		// action cost calculation
		if (feasible1 || feasible2) {
			float cost1 = 0.0f;
			float cost2 = 0.0f;
			for (uint32_t j = 0; j < gp.m; ++j) {
				bool dropj = drop_sh[j * blockDim.x + threadIdx.x];
				uint64_t z1_next = z1;
				uint64_t z2_next = z2;
				if (!dropj) {
					const uint64_t * ptr = go.arrival_transitions_pointers[a * gp.m + j];
					z1_next = ptr[z1 - z_first];
					z2_next = ptr[z2 - z_first];
				}
				if (feasible1) {
					float via_u_z1_next = via_w_old[z1_next] - via_w_old_ref;
					cost1 += tau * gp.lambda[j] * via_u_z1_next;
				}
				if (feasible2) {
					float via_u_z2_next = via_w_old[z2_next] - via_w_old_ref;
					cost2 += tau * gp.lambda[j] * via_u_z2_next;
				}
			}
			if (feasible1 && cost1 < min_action_cost1) {
				min_action_cost1 = cost1;
				best_action1 = a;
			}
			if (feasible2 && cost2 < min_action_cost2) {
				min_action_cost2 = cost2;
				best_action2 = a;
			}
		}
	}
	// update policy
	go.via_policy[z1] = best_action1;
	go.via_policy[z2] = best_action2;
	// update cost
	if (iteration % 2 == 0) {
		go.via_w_1[z1] += min_action_cost1;
		go.via_w_1[z2] += min_action_cost2;
	} else {
		go.via_w_2[z1] += min_action_cost1;
		go.via_w_2[z2] += min_action_cost2;
	}
}

template<bool ignoreRevenue, bool alwaysAllowReject>
__global__ void gpu_optimize_step1_kernel(unsigned iteration, uint64_t z_first) {
	// get values needed to reconstruct u
	float * via_w_old;
	if (iteration % 2 == 0) {
		via_w_old = go.via_w_2;
	} else {
		via_w_old = go.via_w_1;
	}
	float via_w_old_ref = via_w_old[0];
	// compute partial w
	uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t z = z_first + idx;
	if (idx < go.state_batch_size && z < ga.system_state_count) {
		float total_rate = ga.total_arrival_rate + go.departure_rate[idx];
		float via_w_z = 0.0f;
		// server cost
		via_w_z += go.cost[idx];
		// vm revenue
		if (!ignoreRevenue) {
			via_w_z -= go.revenue[idx];
		}
		// future cost for staying in the same state
		float via_u_z = via_w_old[z] - via_w_old_ref;
		via_w_z += (1 - tau * total_rate) * via_u_z;
		// store
		if (iteration % 2 == 0) {
			go.via_w_1[z] = via_w_z;
		} else {
			go.via_w_2[z] = via_w_z;
		}
	}
}

template<bool ignoreRevenue, bool alwaysAllowReject>
__global__ void gpu_optimize_step2_kernel(unsigned iteration, uint64_t z_first) {
	uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t z1 = z_first + idx / 32 * 64 + (idx % 32);
	uint64_t z2 = z1 + 32;
	if ((z2 - z_first) < go.state_batch_size && z2 < ga.system_state_count) {
		gpu_compute_departure_cost_unroll2(z1, z2, iteration, z_first);
	} else if ((z1 - z_first) < go.state_batch_size && z1 < ga.system_state_count) {
		gpu_compute_departure_cost(z1, iteration, z_first);
	}
}

template<bool ignoreRevenue, bool alwaysAllowReject>
__global__ void gpu_optimize_step3_kernel(unsigned iteration, uint64_t z_first) {
	uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t z1 = z_first + idx / 32 * 64 + (idx % 32);
	uint64_t z2 = z1 + 32;
	if ((z2 - z_first) < go.state_batch_size && z2 < ga.system_state_count) {
		gpu_compute_arrival_cost_unroll2<alwaysAllowReject>(z1, z2, iteration, z_first);
	} else if ((z1 - z_first) < go.state_batch_size && z1 < ga.system_state_count) {
		gpu_compute_arrival_cost<alwaysAllowReject>(z1, iteration, z_first);
	}
}

#define CUDA_CHECK(err) if (err != cudaSuccess) {\
	std::stringstream msg; msg << "CUDA Error at line " << __LINE__ << ": " << cudaGetErrorString(err);\
	throw std::runtime_error(msg.str()); }

void gpu_init_state_info(uint64_t z_first) {
	unsigned itemsPerBlock = 128;
	unsigned blocks = (static_cast<unsigned>(cgo.state_batch_size) + itemsPerBlock - 1) /itemsPerBlock;
	gpu_init_state_info_kernel<<<blocks,itemsPerBlock>>>(z_first);
}

float gpu_compute_tau() {
	float max_departure_rate = gpu_reduce_max(static_cast<unsigned>(cgo.state_batch_size), cgo.departure_rate);
	return 0.5f / (cga.total_arrival_rate + max_departure_rate);
}

float gpu_compute_delta_strict() {
	return gpu_reduce_max_absdiff(static_cast<unsigned>(cga.system_state_count), cgo.via_w_1, cgo.via_w_2);
}

void gpu_compute_transitions(uint64_t z_first) {
	unsigned itemsPerBlock = 128;
	unsigned blocks = (static_cast<unsigned>(cgo.state_batch_size) + itemsPerBlock - 1) /itemsPerBlock;
	gpu_compute_transitions_kernel<<<blocks,itemsPerBlock>>>(z_first);
}

void gpu_compute_transition_pointers() {
	unsigned itemsPerBlock = 128;
	unsigned blocks = (static_cast<unsigned>(cga.action_count) + itemsPerBlock - 1) /itemsPerBlock;
	gpu_compute_transition_pointers_kernel<<<blocks,itemsPerBlock>>>();
}

template<bool ignoreRevenue, bool alwaysAllowReject>
void gpu_optimize(unsigned iteration, uint64_t z_first) {
	
	unsigned itemsPerBlock = 128;
	unsigned blocks = (static_cast<unsigned>(cgo.state_batch_size) + itemsPerBlock - 1) /itemsPerBlock;

	gpu_optimize_step1_kernel<ignoreRevenue,alwaysAllowReject><<<blocks,itemsPerBlock>>>(iteration, z_first);
	CUDA_CHECK( cudaDeviceSynchronize() );
	gpu_optimize_step2_kernel<ignoreRevenue,alwaysAllowReject><<<(blocks + 1) / 2,itemsPerBlock>>>(iteration, z_first);
	CUDA_CHECK( cudaDeviceSynchronize() );

	unsigned sh3 = 3 * itemsPerBlock * cgp.m * sizeof(bool);
	gpu_optimize_step3_kernel<ignoreRevenue,alwaysAllowReject><<<(blocks + 1) / 2, itemsPerBlock, sh3>>>(iteration, z_first);
	CUDA_CHECK( cudaDeviceSynchronize() );
}

bool gpu_check_requirements() {
	int dev_count;
	cudaGetDeviceCount(&dev_count);
	if (dev_count == 0) {
		return false;
	}
	int dev;
	cudaGetDevice(&dev);
	cudaDeviceProp dev_prop;
	cudaGetDeviceProperties(&dev_prop, dev);
	if (dev_prop.major == 1 && dev_prop.minor < 1) {
		return false;
	}
	return true;
}

void gpu_optimize_init(const gpu_problem & _cgp , const gpu_analysis & _cga, gpu_optdata & _cgo) {

	int dev;
	cudaDeviceProp dev_prop;
	cudaGetDevice(&dev);
	cudaGetDeviceProperties(&dev_prop, dev);

	size_t free_byte;
	size_t total_byte;

	CUDA_CHECK( cudaMemGetInfo( &free_byte, &total_byte ) );

	std::cout << "Memory: " << (total_byte - free_byte) / (1024*1024) << " MB used out of " << total_byte / (1024*1024) << " MB" << std::endl;

	cgp = _cgp;
	cga = _cga;
	cgo = _cgo;

	CUDA_CHECK( cudaMemcpyToSymbol(gp, &cgp, sizeof(gpu_problem), 0u, cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpyToSymbol(ga, &cga, sizeof(gpu_analysis), 0u, cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpyToSymbol(go, &cgo, sizeof(gpu_optdata), 0u, cudaMemcpyHostToDevice) );

	bool batched = cgo.state_batch_size < cga.system_state_count;

	CUDA_CHECK( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );

	if (!batched) {
		// initialization done only once, for all states
		gpu_init_state_info(0);
		CUDA_CHECK( cudaDeviceSynchronize() );
		gpu_compute_transitions(0);
		CUDA_CHECK( cudaDeviceSynchronize() );
	}

	// shortcut to get action information faster
	gpu_compute_transition_pointers();
	CUDA_CHECK( cudaDeviceSynchronize() );

	if (!batched) {
		ctau = gpu_compute_tau();
		CUDA_CHECK( cudaDeviceSynchronize() );
	} else {
		uint64_t batches = (cga.system_state_count + cgo.state_batch_size - 1) / cgo.state_batch_size;
		float tau = FLT_MAX;
		for (uint64_t b = 0; b  < batches; ++b) {
			uint64_t z_first = b * cgo.state_batch_size;
			gpu_init_state_info(z_first);
			CUDA_CHECK( cudaDeviceSynchronize() );
			tau = std::min(tau, gpu_compute_tau() );
			CUDA_CHECK( cudaDeviceSynchronize() );
		}
		ctau = tau;
	}
	CUDA_CHECK( cudaMemcpyToSymbol(tau, &ctau, sizeof(float), 0u, cudaMemcpyHostToDevice) );

	CUDA_CHECK( cudaMemset(cgo.via_w_1, 0, cga.system_state_count * sizeof(float)) );
}

void gpu_optimize_iteration(bool ignoreRevenue, bool alwaysAllowReject, bool strictConvergence, unsigned iteration, float & cost, float & delta)
{

	bool batched = cgo.state_batch_size < cga.system_state_count;

	if (!batched) {

		if (ignoreRevenue && alwaysAllowReject) {
			gpu_optimize<true,true>(iteration, 0);
		} else if (ignoreRevenue && !alwaysAllowReject) {
			gpu_optimize<true,false>(iteration, 0);
		} else if (!ignoreRevenue && alwaysAllowReject) {
			gpu_optimize<false,true>(iteration, 0);
		} else {
			gpu_optimize<false,false>(iteration, 0);
		}
		CUDA_CHECK( cudaDeviceSynchronize() );

	} else {

		uint64_t batches = (cga.system_state_count + cgo.state_batch_size - 1) / cgo.state_batch_size;

		for (uint64_t b = 0; b < batches; ++b) {
			
			uint64_t z_first = b * cgo.state_batch_size;

			gpu_init_state_info(z_first);
			CUDA_CHECK( cudaDeviceSynchronize() );
	
			gpu_compute_transitions(z_first);
			CUDA_CHECK( cudaDeviceSynchronize() );

			if (ignoreRevenue && alwaysAllowReject) {
				gpu_optimize<true,true>(iteration, z_first);
			} else if (ignoreRevenue && !alwaysAllowReject) {
				gpu_optimize<true,false>(iteration, z_first);
			} else if (!ignoreRevenue && alwaysAllowReject) {
				gpu_optimize<false,true>(iteration, z_first);
			} else {
				gpu_optimize<false,false>(iteration, z_first);
			}

		}

	}

	// get updated cost
	float updated_cost;
	if (iteration % 2 == 0) {
		CUDA_CHECK( cudaMemcpy(&updated_cost, cgo.via_w_1, sizeof(float), cudaMemcpyDeviceToHost) );
	} else {
		CUDA_CHECK( cudaMemcpy(&updated_cost, cgo.via_w_2, sizeof(float), cudaMemcpyDeviceToHost) );
	}

	// compute delta starting from the second iteration
	if (iteration > 1) {
		if (!strictConvergence) {
			delta = std::abs(cost - updated_cost);
		} else {
			delta = gpu_compute_delta_strict();
		}
	}

	// update cost
	cost = updated_cost;
}

void gpu_get_policy(std::vector<uint32_t> & policy) {
	policy.resize(cga.system_state_count);
	CUDA_CHECK( cudaMemcpy(&policy[0], cgo.via_policy, cga.system_state_count * sizeof(uint32_t), cudaMemcpyDeviceToHost) );
}
