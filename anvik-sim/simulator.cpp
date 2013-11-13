#include "simulator.h"

#include <iostream>
#include <limits>

void simulator::simulate(const full_model & fm, const std::vector<uint32_t> & policy) {

	const uint32_t horizon = 10000000;

	const problem & p = fm.get_problem();

	time = 0.0f;
	total_cost = 0.0f;
	total_revenue = 0.0f;

	// get initial (idle) state
	system_state_idx = 0;
	fm.get_system_state_info(system_state_idx, system_state_info);

	// initialize random number generation
	rng = std::mt19937();
	arrival_time_dists.resize(p.m);
	for (uint32_t j = 0; j < p.m; ++j) {
		arrival_time_dists[j] = std::exponential_distribution<double>(p.lambda[j]);
	}

	// initialize next departure ids
	current_departure_ids.resize(p.k);
	for (uint32_t i = 0; i < p.k; ++i) {
		current_departure_ids[i].resize(fm.get_server_states(i));
	}

	// generate first event
	for (uint32_t j = 0; j < p.m; ++j) {
		schedule_next_arrival(j);
	}

	for (uint64_t q = 0; q < horizon; ++q) {
		bool has_departure = false;
		while (!departures.empty()) {
			const departure_event & e = departures.top();
			if (e.id == current_departure_ids[e.server_group][e.server_state]) {
				has_departure = true;
				break;
			} else {
				departures.pop();
			}
		}
		// decide if next event is an arrival or a departure
		if (!has_departure || arrivals.top().time < departures.top().time) {
			// arrival
			const arrival_event & next = arrivals.top();
			// update time and total cost
			double elapsed = next.time - time;
			total_cost += elapsed / next.time * (system_state_info.cost - total_cost);
			total_revenue += elapsed / next.time * (system_state_info.revenue - total_revenue);
			time = next.time;
			// change state and schedule departure
			change_state(next, fm, policy);
			// schedule next arrival
			schedule_next_arrival(next.vm_class);
			// remove event
			arrivals.pop();
		} else {
			// departure
			const departure_event & next = departures.top();
			// update time and total cost
			double elapsed = next.time - time;
			total_cost += elapsed / next.time * (system_state_info.cost - total_cost);
			total_revenue += elapsed / next.time * (system_state_info.revenue - total_revenue);
			time = next.time;
			// change state
			change_state(next, fm);
			// remove envent
			departures.pop();
		}
	}
}

void simulator::schedule_next_arrival(uint32_t j) {
	arrival_event e;
	e.vm_class = j;
	e.time = time + arrival_time_dists[j](rng);
	arrivals.push(e);
}

void simulator::schedule_next_departure(const full_model & fm, uint32_t i, uint32_t w) {
	// update id of departure event from servers of group i in state w
	current_departure_ids[i][w] += 1;
	// genearte new departure event if there is at least one server
	uint32_t servers = system_state_info.group_states[i][w];
	if (servers > 0) {
		const problem & p = fm.get_problem();
		bool next = false;
		uint32_t j_next;
		double time_next = std::numeric_limits<double>::infinity();
		for (uint32_t j = 0; j < p.m; ++j) {
			uint32_t vms = fm.get_server_state(i, w)[j];
			if (vms > 0 && p.mu[j] > 0) {
				double rate = servers * vms * p.mu[j];
				std::exponential_distribution<double> dist(rate);
				double time = dist(rng);
				if (time < time_next) {
					next = true;
					j_next = j;
					time_next = time;
				}
			}
		}
		if (next) {
			departure_event e;
			e.server_group = i;
			e.server_state = w;
			e.vm_class = j_next;
			e.time = time + time_next;
			e.id = current_departure_ids[i][w];
			departures.push(e);
		}
	}
}

void simulator::change_state(const arrival_event & e, const full_model & fm, const std::vector<uint32_t> & policy) {
	uint32_t a = policy[system_state_idx];
	const full_model::action & action = fm.get_action(a)[e.vm_class];
	if (!action.drop) {
		uint32_t j = e.vm_class;
		uint32_t i = action.server_type;
		uint32_t w = action.server_alloc;
		// get next server state index
		uint32_t w_next = fm.get_arrival_transitions(i, w)[j];
		// compute next group state index
		uint64_t zi_next = fm.find_next_group_state(i, system_state_info.state[i], system_state_info.group_states[i], w, w_next);
		// compute next system state index
		system_state_idx = fm.find_next_system_state(system_state_idx,  system_state_info.state, i, zi_next);
		// get next system state
		fm.get_system_state_info(system_state_idx, system_state_info);
		// thanks to memory less propert we can schedule next departure
		schedule_next_departure(fm, i, w);
		schedule_next_departure(fm, i, w_next);
	}
}

void simulator::change_state(const departure_event & e, const full_model & fm) {
	uint32_t j = e.vm_class;
	uint32_t i = e.server_group;
	uint32_t w = e.server_state;
	uint32_t w_next = fm.get_departure_transitions(i, w)[j];
	// compute next group state index
	uint64_t zi_next = fm.find_next_group_state(i, system_state_info.state[i], system_state_info.group_states[i], w, w_next);
	// compute next system state index
	system_state_idx = fm.find_next_system_state(system_state_idx,  system_state_info.state, i, zi_next);
	// get next system state
	fm.get_system_state_info(system_state_idx, system_state_info);
	// thanks to memory less propert we can schedule next departure
	schedule_next_departure(fm, i, w);
	schedule_next_departure(fm, i, w_next);
}