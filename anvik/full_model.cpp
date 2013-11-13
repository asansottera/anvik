#include "full_model.h"
#include "print_utilities.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <cassert>

void full_model::enumerate_feasible_allocations_recursive(
	uint32_t i,
	const std::vector<uint32_t> & allocation,
	const std::vector<uint32_t> & used,
	uint32_t level) {
	if (level < p.m) {
		std::vector<uint32_t> my_allocation = allocation;
		std::vector<uint32_t> my_used = used;
		bool full = false;
		while (!full) {
			// descend
			enumerate_feasible_allocations_recursive(i, my_allocation, my_used, level+1);
			// increase VMs of this type
			my_allocation[level] += 1;
			// increase used resources
			for (uint32_t h = 0; h < p.r; ++h) {
				my_used[h] += p.l[level][h];
			}
			// check if we can further increase the number of VMs of this type
			for (uint32_t h = 0; h < p.r; ++h) {
				if (my_used[h] > p.c[i][h]) {
					full = true;
					break;
				}
			}
		}
	} else {
		feasible_allocations[i].push_back(allocation);
	}
}

void full_model::enumerate_feasible_allocations(uint32_t i) {
	std::vector<uint32_t> partial(p.m);
	std::vector<uint32_t> used(p.r);
	enumerate_feasible_allocations_recursive(i, partial, used, 0);
}

void full_model::enumerate_feasible_allocations() {
	feasible_allocations.resize(p.k);
	for (uint32_t i = 0; i < p.k; ++i) {
		enumerate_feasible_allocations(i);
		std::cout << "Feasible allocations on servers of type " << i << ": " << feasible_allocations[i].size() << std::endl;
		for (uint32_t w = 0; w < feasible_allocations[i].size(); ++w) {
			std::cout << "\t" << w << ": " << feasible_allocations[i][w] << std::endl;
		}
	}
}

void full_model::enumerate_states() {
	states.resize(p.k);
	for (uint32_t i = 0; i < p.k; ++i) {
		states[i].init(p.n[i], static_cast<uint32_t>(feasible_allocations[i].size()));
	}
}

void full_model::get_system_state_info(uint64_t idx, state_info & info) const {
	// initialize
	info.state.resize(p.k);
	info.accept_vm.resize(p.m);
	info.running_vm.resize(p.m);
	info.group_states.resize(p.k);
	for (uint32_t i = 0; i < p.k; ++i) {
		info.group_states[i].resize(get_server_states(i));
	}
	std::fill(info.accept_vm.begin(), info.accept_vm.end(), false);
	std::fill(info.running_vm.begin(), info.running_vm.end(), 0);
	info.cost = 0;
	// get state vector
	get_system_state(idx, info.state);
	// compute number of VMs and whether new VMs are accepted or not
	for (uint32_t i = 0; i < p.k; ++i) {
		uint64_t num_allocs = feasible_allocations[i].size();
		// get state for servers of type i
		states[i].get(info.state[i], info.group_states[i]);
		for (uint32_t w = 0; w < num_allocs; ++w) {
			uint32_t count = info.group_states[i][w];
			if (count > 0) {
				const std::vector<uint32_t> & feas_alloc_iw = feasible_allocations[i][w];
				const std::vector<uint32_t> & arr_tran_iw = arrival_transitions[i][w];
				for (uint32_t j = 0; j < p.m; ++j) {
					info.running_vm[j] += count * feas_alloc_iw[j];
					// if neededed, check whether arrivals of class j are accepted
					if (!info.accept_vm[j]) {
						// check if servers of type i with allocation w
						// can accept a new arrival of class j
						if (arr_tran_iw[j] != w) {
							info.accept_vm[j] = true;
						}
					}
				}
			}
		}
	}
	// compute other information
	info.cost = compute_state_cost(info);
	info.revenue = compute_state_revenue(info);
	info.departure_rate = compute_state_departure_rate(info);
}

void full_model::get_group_state(uint32_t i, uint64_t idx, std::vector<uint32_t> & state) const {
	states[i].get(idx, state);
}

uint64_t full_model::find_group_state(uint32_t i, const std::vector<uint32_t> & state) const {
	return states[i].find(state);
}

void full_model::evaluate_allowed_arrivals(uint64_t z, std::vector<bool> & allowed) {
	allowed.resize(p.m);
	std::vector<uint64_t> state(p.k);
	get_system_state(z, state);
	// for each arrival class
	for (uint32_t j = 0; j < p.m; ++j) {
		// for each type of servers
		for (uint32_t i = 0; i < p.k; ++i) {
			uint64_t num_allocs_i = feasible_allocations[i].size();
			// get state for ervers of type i
			std::vector<uint32_t> istate(num_allocs_i);
			states[i].get(state[i], istate);
			// for each allocation of servers of type i
			for (uint32_t w = 0; w < num_allocs_i; ++w) {
				uint64_t servers_i_w = istate[w];
				if (servers_i_w > 0 && arrival_transitions[i][w][j] != w) {
					allowed[j] = true;
					break;
				}
			}
			if (allowed[j]) {
				break;
			}
		}
	}
}

uint64_t full_model::find_system_state(const std::vector<uint64_t> & state) const {
	uint64_t idx = 0;
	uint64_t factor = 1;
	for (uint32_t i = 0; i < p.k; ++i) {
		idx += state[i] * factor;
		factor *= states[i].size();
	}
	return idx;
}

void full_model::get_system_state(uint64_t idx, std::vector<uint64_t> & state) const {
	uint64_t remainder = idx;
	uint64_t factor = 1;
	for (uint32_t i = 0; i < p.k; ++i) {
		factor *= states[i].size();
	}
	for (uint32_t i = 0; i < p.k; ++i) {
		factor /= states[p.k-i-1].size();
		state[p.k-i-1] =  remainder / factor;
		remainder = remainder % factor;
	}
	assert(find_system_state(state) == idx);
}

void full_model::enumerate_arrival_transitions() {
	arrival_transitions.resize(p.k);
	// for each server type
	for (uint32_t i = 0; i < p.k; ++i) {
		std::cout << "Transitions for servers of type " << i << " on arrivals" << std::endl;
		arrival_transitions[i].resize(feasible_allocations[i].size());
		// for each feasible allocations on servers of type i
		for (uint32_t w = 0; w < feasible_allocations[i].size(); ++w) {
			arrival_transitions[i][w].resize(p.m);
			// compute usage with this allocation
			std::vector<uint32_t> usage(p.r);
			for (uint32_t j = 0; j < p.m; ++j) {
				for (uint32_t h = 0; h < p.r; ++h) {
					usage[h] += p.l[j][h] * feasible_allocations[i][w][j];
				}
			}
			// compute transition on arrival of class j
			for (uint32_t j = 0; j < p.m; ++j) {
				// check if there is space for an arrival of class j
				bool reject = false;
				for (uint32_t h = 0; h < p.r; ++h) {
					if (usage[h] + p.l[j][h] > p.c[i][h]) {
						reject = true;
						break;
					}
				}
				if (reject) {
					// infeasible
					arrival_transitions[i][w][j] = w;
				} else {
					// find allocation after arrival
					std::vector<uint32_t> alloc_to_find = feasible_allocations[i][w];
					alloc_to_find[j] += 1;
					uint32_t w2 = find_allocation(i, alloc_to_find);
					arrival_transitions[i][w][j] = w2;
					std::cout << "\tfrom ";
					std::cout << feasible_allocations[i][w] << " to " << feasible_allocations[i][w2];
					std::cout << " on arrivals of class " << j << std::endl;
				}
			}
		}
	}
}

void full_model::enumerate_departure_transitions() {
	departure_transitions.resize(p.k);
	// for each server type
	for (uint32_t i = 0; i < p.k; ++i) {
		std::cout << "Transitions for servers of type " << i << " on departures" << std::endl;
		departure_transitions[i].resize(feasible_allocations[i].size());
		// for each feasible allocations on servers of type i
		for (uint32_t w = 0; w < feasible_allocations[i].size(); ++w) {
			departure_transitions[i][w].resize(p.m);
			// compute transition on departure of class j
			for (uint32_t j = 0; j < p.m; ++j) {
				if (feasible_allocations[i][w][j] == 0) {
					// infeasible
					departure_transitions[i][w][j] = w;
				} else {
					std::vector<uint32_t> alloc_to_find = feasible_allocations[i][w];
					alloc_to_find[j] -= 1;
					uint32_t w2 = find_allocation(i, alloc_to_find);
					departure_transitions[i][w][j] = w2;
					std::cout << "\tfrom ";
					std::cout << feasible_allocations[i][w] << " to " << feasible_allocations[i][w2];
					std::cout << " on departures of class " << j << std::endl;
				}
			}
		}
	}
}

uint32_t full_model::find_allocation(uint32_t i, const std::vector<uint32_t> & alloc) {
	for (uint32_t w2 = 0; w2 < feasible_allocations[i].size(); ++w2) {
		bool match = true;
		for (uint32_t j2 = 0; j2 < p.m; ++j2) {
			if (feasible_allocations[i][w2][j2] != alloc[j2]) {
				match = false;
				break;
			}
		}
		if (match) {
			return w2;
		}
	}
	throw std::runtime_error("Allocation not found");
}

void full_model::enumerate_actions() {
	// count total number of actions
	uint64_t num_actions_per_class = 1;
	for (uint32_t i = 0; i < p.k; ++i) {
		num_actions_per_class += feasible_allocations[i].size();
	}
	uint64_t num_actions = static_cast<uint64_t>(std::pow(num_actions_per_class, p.m));
	std::cout << "Actions: " << num_actions << std::endl;
	// allocate memory
	actions.reserve(num_actions);
	// use queue to implement depth first enumeration
	typedef std::vector<action> vaction;
	std::queue<std::pair<vaction,uint32_t>> action_queue;
	vaction current(p.m);
	uint32_t level = 0;
	action_queue.push(std::pair<vaction,uint32_t>(current,level));
	while (!action_queue.empty()) {
		current = action_queue.front().first;
		level = action_queue.front().second;
		action_queue.pop();
		if (level < p.m) {
			current[level] = action::drop_action();
			action_queue.push(std::pair<vaction,uint32_t>(current, level+1));
			for (uint32_t i = 0; i < p.k; ++i) {
				for (uint32_t w = 0; w < feasible_allocations[i].size(); ++w) {
					current[level] = action(i,w);
					action_queue.push(std::pair<vaction,uint32_t>(current, level+1));
				}
			}
		} else {
			actions.push_back(current);
		}
	}
}

float full_model::compute_state_cost(const state_info & info) const {
	float cost = 0.0f;
	for (uint32_t i = 0; i < p.k; ++i) {
		// state of servers of type i
		uint64_t y = info.state[i];
		// get number of active servers of type i
		uint32_t active = p.n[i] - info.group_states[i][0];
		// compute cost
		cost += active * p.cost[i];
	}
	return cost;
}

float full_model::compute_state_revenue(const state_info & info) const {
	float revenue = 0.0f;
	for (uint32_t j = 0; j < p.m; ++j) {
		revenue += info.running_vm[j] * p.revenue[j];
	}
	return revenue;
}

float full_model::compute_state_arrival_rate(const state_info & info) const {
	float arrival_rate = 0.0f;
	for (uint32_t j = 0; j < p.m; ++j) {
		// arrival rate for VMs of class j
		if (info.accept_vm[j]) {
			arrival_rate += p.lambda[j];
		}
		// next class
	}
	return arrival_rate;
}

float full_model::compute_state_departure_rate(const state_info & info) const {
	float departure_rate = 0.0f;
	for (uint32_t j = 0; j < p.m; ++j) {
		uint32_t count = info.running_vm[j];
		departure_rate += count * p.mu[j];
	}
	return departure_rate;
}

void full_model::print_system_state(uint64_t z) {
    std::vector<uint64_t> state(p.k);
    get_system_state(z, state);
    std::cout << "System state " << z << std::endl;
    for (uint32_t i = 0; i < p.k; ++i) {
        std::vector<uint32_t> state_i;
        states[i].get(state[i], state_i);
        std::cout << "\tGroup " << i << " in state " << state[i] << ": ";
        std::cout << state_i << std::endl;
    }
}

void full_model::print_action(uint32_t a) {
	for (uint32_t j = 0; j < p.m; ++j) {
		std::cout << "\tVM of class " << j << ": ";
		if (actions[a][j].drop) {
			std::cout << "drop" << std::endl;
		} else {
			uint32_t i = actions[a][j].server_type;
			uint32_t w = actions[a][j].server_alloc;
			std::cout << "server of type " << i << " with allocation " << w << std::endl;
		}
	}
}

full_model::full_model() : analysis(false) {
}

full_model::full_model(const problem & _p) : analysis(false) {
	analyze(_p);
}

#define ANVIK_FULL_MODEL_VERBOSITY 0

void full_model::analyze(const problem & _p) {
	_p.check();
	// copy problem
	p = _p;
	// for each server type, the vectors with all the feasible allocations
	enumerate_feasible_allocations();
	// for each server type, enumerate transitions from one allocation to another
	enumerate_arrival_transitions();
	enumerate_departure_transitions();
	// count number of states
	// enumerate states
	enumerate_states();
	state_space_size = 1;
	for (uint32_t i = 0; i < p.k; ++i) {
		uint64_t i_size = states[i].size();
		if (std::numeric_limits<uint64_t>::max() / state_space_size < i_size) {
			throw std::runtime_error("Too many system states: uint64_t overflow");
		}
		state_space_size *= states[i].size();
	}
	std::cout << "States: " << state_space_size << std::endl;
    for (uint32_t i = 0; i < p.k; ++i) {
        std::cout << "\tStates of group " << i << ": " << states[i].size() << std::endl;
    }
	// show some example states
	#if ANVIK_FULL_MODEL_VERBOSITY >= 1
	std::cout << "Showing reference state: " << std::endl;
	{
		std::vector<uint64_t> state(p.k, 0);
		uint64_t idx = find_system_state(state);
		std::cout << "\tState " << idx << ": " << std::endl;
		for (uint32_t i = 0; i < p.k; ++i) {
			std::vector<uint32_t> state_i;
			states[i].get(state[i], state_i);
			std::cout << "\t\tServers of type " << i << " are in state " << state[i] << ": " << state_i << std::endl;
		}
	}
	std::cout << "Showing a few random states..." << std::endl;
	for (uint32_t test = 0; test < 10; ++test) {
		std::vector<uint64_t> state(p.k);
		for (uint32_t i = 0; i < p.k; ++i) {
			state[i] = std::rand() % states[i].size();
		}
		uint64_t idx = find_system_state(state);
		std::cout << "\tState " << idx << ": " << std::endl;
		for (uint32_t i = 0; i < p.k; ++i) {
			std::vector<uint32_t> state_i;
			states[i].get(state[i], state_i);
			std::cout << "\t\tServers of type " << i << " are in state " << state[i] << ": " << state_i << std::endl;
		}
	}
	#endif
	// enumerate actions
	enumerate_actions();
	#if ANVIK_FULL_MODEL_VERBOSITY >= 1
	for (uint32_t a = 0; a < actions.size(); ++a) {
		std::cout << "Action " << a << ": " << std::endl;
		print_action(a);
	}
	#endif
	// mark analysis completed
	analysis = true;
}

uint64_t full_model::find_next_system_state(
	const uint64_t z,
	const std::vector<uint64_t> & state,
	const uint32_t i,
	const uint64_t next_state_i) const
{
	uint64_t factor = 1;
	for (uint32_t i2 = 0; i2 < i; ++i2) {
		factor *= get_group_states(i2);
	}
	uint64_t z_next = z + (next_state_i - state[i]) * factor;
	return z_next;
}

uint64_t full_model::find_next_group_state(
	const uint32_t i,
	const uint64_t zi,
	const std::vector<uint32_t> & group_state,
	const uint32_t w,
	const uint32_t w_next) const
{
	const multichoose & mc = states[i].multichoose_table();
	// number of servers of type i
	uint32_t N = p.n[i];
	// number of possible states for servers of type i
	uint32_t W = get_server_states(i);
	// compute next state
	uint64_t zi_next = zi;
	if (w < w_next) {
		uint32_t remaining = N;
		for (uint32_t w2 = 0; w2 < w; ++w2) {
			remaining -= group_state[w2];
		}
		for (uint32_t w2 = w; w2 < w_next; ++w2) {
			remaining -= group_state[w2];
			zi_next += mc(remaining, W-w2-1);
		}
	} else if (w > w_next) {
		uint32_t remaining = N;
		for (uint32_t w2 = 0; w2 < w_next; ++w2) {
			remaining = remaining-group_state[w2];
		}
		for (uint32_t w2 = w_next; w2 < w; ++w2) {
			remaining -= group_state[w2];
			zi_next -= mc(remaining - 1, W-w2-1);
		}
	}
	return zi_next;
}