#include "cpu_optimizer.h"

#include <iostream>

#include <thread>
#include <atomic>
#include <chrono>
#include <system_error>

struct worker {
	std::thread thread;
	std::atomic<bool> cancel;
	std::atomic<uint32_t> iteration;
	worker() : cancel(false) {
	}
};

template<class TFloat>
cpu_optimizer<TFloat>::cpu_optimizer() {
	w = new worker();
}

template<class TFloat>
cpu_optimizer<TFloat>::~cpu_optimizer() {
	delete w;
}

template<class TFloat>
bool cpu_optimizer<TFloat>::is_available() const {
	// always available
	return true;
}

template<class TFloat>
uint32_t cpu_optimizer<TFloat>::get_iteration() const {
	return w->iteration;
}

template<class TFloat>
void cpu_optimizer<TFloat>::optimize(const full_model & fm) {
	// set threshold
	const TFloat delta_threshold = 1e-4f;
	// get information
	const problem & p = fm.get_problem();
	uint64_t state_space_size = fm.get_system_states();
	uint64_t action_space_size = fm.get_actions();
	// start timing
	std::chrono::high_resolution_clock::time_point start, end;
	start = std::chrono::high_resolution_clock::now();
	// compute total arrival rate
	TFloat total_arrival_rate = 0.0f;
	for (uint32_t j = 0; j < p.m; ++j) {
		total_arrival_rate += p.lambda[j];
	}
	// compute the constant tau
	TFloat tau = compute_tau<TFloat>(fm);
	// compute optimal policy through value iteration algorithm
	via_w_1.resize(state_space_size);
	via_w_2.resize(state_space_size);
	policy.resize(state_space_size);
	// initialize iteration counter
	unsigned iteration = 0;
	TFloat delta = 1.0f;
	while (delta > delta_threshold && ! w->cancel) {
		// update iteration counter
		++iteration;
		// write atomic variable for other threads (e.g., GUI thread)
		w->iteration = iteration;
		// run iteration
		# pragma omp parallel
		{
			// objects that hold state information
			// they are all allocated here to avoid malloc in the critical path
			full_model::state_info info;
			std::vector<uint64_t> state_next(p.k);
			std::vector<matrix<uint64_t>> arrival_next_state(p.k);
			for (uint32_t i = 0; i < p.k; ++i) {
				arrival_next_state[i].resize(p.m, fm.get_server_states(i));
			}
			# pragma omp for schedule(dynamic,64) nowait
			for (int64_t z = 0; z < static_cast<int64_t>(state_space_size); ++z) {
				optimize_state_iteration(
					fm, z,
					info, arrival_next_state,
					total_arrival_rate, tau, iteration);
			}
		}
		// check exit condition
		if (iteration > 1) {
			if (!this->check_strict_convergence) {
				delta = std::abs(via_w_1[0] - via_w_2[0]);
			} else {
				delta = 0.0f;
				#pragma omp parallel
				{
					TFloat delta_private = 0.0f;
					# pragma omp for schedule(static,32) nowait
					for (int64_t z = 0; z < static_cast<int64_t>(state_space_size); ++z) {
						delta_private = std::max(delta, std::abs(via_w_1[z] - via_w_2[z]));
					}
					# pragma omp critical
					{
						delta = std::max(delta, delta_private);
					}
				}
			}
		}
		if (iteration % 2 == 0) {
			objective = via_w_1[0];
		} else {
			objective = via_w_2[0];
		}
		std::cout << "Iteration " << iteration << ": delta = " << delta << ", objective = " << objective << std::endl;
	}
	// release memory
	via_w_1.clear();
	via_w_2.clear();
	// end timing
	end = std::chrono::high_resolution_clock::now();
	long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
	// report information
	if (delta < delta_threshold) {
		std::cout << "Value Iteration Algorithm completed in " << iteration << " iterations" << std::endl;
	} else {
		std::cout << "Value Iteration Algorithm canceled after " << iteration << " iterations" << std::endl;
	}
	std::cout << "Value Iteration Algorithm run for " << ms << " ms" << std::endl;
	std::cout << "Value Iteration Algorithm: average cost: " << objective << std::endl;
}

template<class TFloat>
void cpu_optimizer<TFloat>::optimize_state_iteration(
	const full_model & fm,
	const uint64_t z,
	full_model::state_info & info,
	std::vector<matrix<uint64_t>> & arrival_next_state,
	const TFloat total_arrival_rate,
	const TFloat tau, 
	const unsigned iteration)
{
	// determine old w vector to compute u value
	const TFloat * via_w_old;
	if (iteration % 2 == 0) {
		via_w_old = &via_w_2[0];
	} else {
		via_w_old = &via_w_1[0];
	}
	TFloat via_w_ref_old = via_w_old[0];
	// temporary value
	TFloat via_w_z = 0.0f;
	// get information
	const problem & p = fm.get_problem();
	uint64_t action_space_size = fm.get_actions();
	// compute information for state z
	fm.get_system_state_info(z, info);
	TFloat total_rate = 0.0f;
	total_rate += total_arrival_rate;
	total_rate += info.departure_rate;
	// server cost and VM revenue
	via_w_z += info.cost;
	if (!ignore_revenue) {
		via_w_z -= info.revenue;
	}
	// cost for staying in the same state (no event occurred)
	via_w_z += (1 - tau * total_rate) * (via_w_old[z] - via_w_ref_old);
	// future cost associated with departure transitions
	for (uint32_t i = 0; i < p.k; ++i) {
		uint64_t server_states = fm.get_server_states(i);
		// for each allocation (server state)
		for (uint32_t w  = 0; w < server_states; ++w) {
			uint32_t servers = info.group_states[i][w];
			if (servers > 0) {
				const std::vector<uint32_t> & allocation = fm.get_server_state(i, w);
				// number of servers of type i in state w
				for (uint32_t j = 0; j < p.m; ++j) {
					if (allocation[j] > 0) {
						// rate of departures of VMs of type j
						// from a server of type i in allocation w
						TFloat rate = servers * allocation[j] * p.mu[j];
						// next allocation
						uint32_t w_next = fm.get_departure_transitions(i,w)[j];
						// find index of next group state
						uint64_t zi_next = fm.find_next_group_state(i, info.state[i], info.group_states[i], w, w_next);
						// find index of next system state
						uint64_t z_next = fm.find_next_system_state(z, info.state, i, zi_next);
						// update w
						TFloat via_u_z_next = via_w_old[z_next] - via_w_ref_old;
						TFloat tmp = tau * rate * via_u_z_next;
						via_w_z += tmp;
					}
				}
			}
		}
	}
	// pre-compute arrival next states
	for (uint32_t i = 0; i < p.k; ++i) {
		uint64_t server_states = fm.get_server_states(i);
		for (uint32_t w = 0; w < server_states; ++w) {
			uint32_t servers = info.group_states[i][w];
			// if feasible...
			if (servers > 0) {
				for (uint32_t j = 0; j < p.m; ++j) {
					uint32_t w_next = fm.get_arrival_transitions(i,w)[j];
					if (w != w_next) {
						// find index of next group state
						uint64_t zi_next = fm.find_next_group_state(i, info.state[i], info.group_states[i], w, w_next);
						// find index of next system state
						uint64_t z_next = fm.find_next_system_state(z, info.state, i, zi_next);
						// store index of next system state
						arrival_next_state[i](j,w) = z_next;
					} else {
						// system state remains the same
						arrival_next_state[i](j,w) = z;
					}
				}
			}
		}
	}
	// compute best action
	TFloat min_action_cost = std::numeric_limits<TFloat>::infinity();
	uint32_t best_action = 0;
	for (uint32_t a = 0; a < action_space_size; ++a) {
		const std::vector<full_model::action> & actionv = fm.get_action(a);
        // determine if action is feasible
		bool action_feasible = true;
        for (uint32_t j = 0; j < p.m; ++j) {
			if (info.accept_vm[j] && !actionv[j].drop) {
 				uint32_t i = actionv[j].server_type;
				uint32_t w = actionv[j].server_alloc;
				uint32_t w_next = fm.get_arrival_transitions(i,w)[j];
				bool feasible = (w_next != w) && info.group_states[i][w] > 0;
				if (!feasible) {
                    action_feasible = false;
                    break;
                }
            } else if (info.accept_vm[j] && actionv[j].drop && !always_allow_reject) {
                action_feasible = false;
                break;
            } else if (!info.accept_vm[j] && !actionv[j].drop) {
                action_feasible = false;
                break;
            }
        }
        // compute action cost
        if (action_feasible) {
            TFloat action_cost = 0.0f;
            for (uint32_t j = 0; j < p.m; ++j) {
                if (!actionv[j].drop) {
                    uint32_t i = actionv[j].server_type;
                    uint32_t w = actionv[j].server_alloc;
					// get index of next system state
					uint64_t z_next = arrival_next_state[i](j,w);
					// update w
					TFloat via_u_z_next = via_w_old[z_next] - via_w_ref_old;
                    action_cost += tau * p.lambda[j] * via_u_z_next;
                } else {
					// update w
					TFloat via_u_z = via_w_old[z] - via_w_ref_old;
					action_cost += tau * p.lambda[j] * via_u_z;
				}
            }
            // update best action
            if (action_cost < min_action_cost) {
                min_action_cost = action_cost;
                best_action = a;
            }
        }
	}
	via_w_z += min_action_cost;
	// update data
	if (iteration % 2 == 0) {
		via_w_1[z] = via_w_z;
	} else {
		via_w_2[z] = via_w_z;
	}
	policy[z] = best_action;
}

template<class TFloat>
void cpu_optimizer<TFloat>::start_optimize(const full_model & fm) {
	w->iteration = 0;
	w->cancel = false;
	w->thread = std::thread(&cpu_optimizer<TFloat>::optimize, this, std::cref(fm));
}

template<class TFloat>
void cpu_optimizer<TFloat>::cancel_optimize() {
	w->cancel = true;
}

template<class TFloat>
void cpu_optimizer<TFloat>::join_optimize() {
	try {
		w->thread.join();
	} catch (std::system_error & e) {
		std::cout << e.what() << std::endl;
	}
}

template<class TFloat>
TFloat compute_tau(const full_model & fm) {
	const problem & p = fm.get_problem();
	uint64_t state_space_size = fm.get_system_states();
	TFloat total_arrival_rate = 0.0f;
	for (uint32_t j = 0; j < p.m; ++j) {
		total_arrival_rate += p.lambda[j];
	}
	TFloat max_rate = 0.0f;
	#pragma omp parallel
	{
		full_model::state_info info;
		TFloat max_rate_t = 0.0f;
		#pragma omp for schedule(static,128) nowait
		for (int64_t z = 0; z < static_cast<int64_t>(state_space_size); ++z) {
			fm.get_system_state_info(z, info);
			max_rate_t = std::max<TFloat>(max_rate_t, info.departure_rate);
		}
		#pragma omp critical
		{
			max_rate = std::max(max_rate, max_rate_t);
		}
	}
	max_rate += total_arrival_rate;
	TFloat tau = 0.5f / max_rate;
	return tau;
}

template class cpu_optimizer<float>;
template class cpu_optimizer<double>;
template float compute_tau<float>(const full_model & fm);
template double compute_tau<double>(const full_model & fm);
