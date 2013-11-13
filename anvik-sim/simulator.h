#ifndef ANVIK_SIMULATOR_H
#define ANVIK_SIMULATOR_H

#include "problem.h"
#include "full_model.h"
#include <cstdint>
#include <queue>
#include <random>
#include <vector>

class simulator {
public:
	void simulate(const full_model & fm, const std::vector<uint32_t> & policy);
	double get_time() const {
		return time;
	}
	double get_average_cost() const {
		return total_cost;
	}
	double get_average_revenue() const {
		return total_revenue;
	}
	double get_average_loss() const {
		return (total_cost - total_revenue);
	}
	double get_average_profit() const {
		return (total_revenue - total_cost);
	}
private:
	struct arrival_event {
		uint32_t vm_class;
		double time;
		bool operator<(const arrival_event & e2) const {
			return time > e2.time;
		}
	};
	struct departure_event {
		uint32_t server_group;
		uint32_t server_state;
		uint32_t vm_class;
		double time;
		uint32_t id;
		bool operator<(const departure_event & e2) const {
			return time > e2.time;
		}
	};
	typedef std::priority_queue<arrival_event> arrival_queue;
	typedef std::priority_queue<departure_event> departure_queue;
private:
	void schedule_next_arrival(uint32_t j);
	void schedule_next_departure(const full_model & fm, uint32_t i, uint32_t w);
	void change_state(const arrival_event & e, const full_model & fm, const std::vector<uint32_t> & policy);
	void change_state(const departure_event & e, const full_model & fm);
private:
	std::vector<std::vector<uint32_t>> current_departure_ids;
	double time;
	double total_cost;
	double total_revenue;
	uint64_t system_state_idx;
	full_model::state_info system_state_info;
	arrival_queue arrivals;
	departure_queue departures;
	std::mt19937 rng;
	std::vector<std::exponential_distribution<double>> arrival_time_dists;
};

#endif