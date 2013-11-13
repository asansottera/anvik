#ifndef ANVIK_FULL_MODEL_H
#define ANVIK_FULL_MODEL_H

#include "problem.h"
#include "combinatorics.h"
#include "matrix.h"
#include "comb_with_rep.h"

class full_model {
public:
	full_model();
	explicit full_model(const problem & _p);
	/* Performs the analysis (it is mandatory to call this method before the optimization. */
	void analyze(const problem & _p);
	/* Solve with ignore_revenue = true and allow_unrequired_drop = false. */
	void optimize();
	/* Solve with the specified settings. */
	void optimize(bool ignore_revenue, bool allow_unrequired_drop);
public:
	/* Returns a const reference to the problem object. */
	const problem & get_problem() const {
		return p;
	}
public:
	/* Returns the number of states for a server of type i. */
	uint32_t get_server_states(uint32_t i) const {
		return static_cast<uint32_t>(feasible_allocations[i].size());
	}
	/* Returns the number of states for the i-th group of server. */
	uint64_t get_group_states(uint32_t i) const {
		return states[i].size();
	}
	/* Returns the overall number of states. */
	uint64_t get_system_states() const {
		return state_space_size;
	}
	/* Returns the number of actions. */
	uint64_t get_actions() const {
		return actions.size();
	}
public:
	struct state_info {
		std::vector<uint64_t> state;
		std::vector<std::vector<uint32_t>> group_states;
		std::vector<uint8_t> accept_vm;
		std::vector<uint32_t> running_vm;
		float cost;
		float revenue;
		float departure_rate;
	};
	struct action {
		bool drop;
		uint32_t server_type;
		uint32_t server_alloc;
		action() {
		}
		action(uint32_t i, uint32_t w) : drop(false), server_type(i), server_alloc(w) {
		}
		static action drop_action() {
			action a;
			a.drop = true;
			return a;
		}
	};
public:
	/* Fills the vector with the system state with the given index.
	   The i-th element of the vector is the state of servers of type i. */
	void get_system_state(uint64_t idx, std::vector<uint64_t> & state) const;
	/* Gets information about the system state with the given index. */
	void get_system_state_info(uint64_t idx, state_info & info) const;
	/* Finds the index of a system state.
	   The i-th element of the vector is the state of servers of type i.  */
	uint64_t find_system_state(const std::vector<uint64_t> & state) const; 
	/* Gets a state of the i-th group of servers, given the index.
	   The w-th element of the vector is the number of server in server state w.*/
	void get_group_state(uint32_t i, uint64_t idx, std::vector<uint32_t> & state) const;
	/* Finds the index of a state of the i-th group of servers.
	   The w-th element of the vector is the number of server in server state w. */
	uint64_t find_group_state(uint32_t i, const std::vector<uint32_t> & state) const;
public:
	/* Returns a const reference to the a-th action.
	   The j-th elemnt of the vector is the action to take for arrivals of class j. */
	const std::vector<action> & get_action(uint64_t a) const {
		return actions[a];
	}
	/* Returns a const reference to the w-th feasible server state (allocation) for serers of type i.
	   The j-th element of the vector is the number of VMs of class j allocated on the server. */
	const std::vector<uint32_t> & get_server_state(uint32_t i, uint32_t w) const {
		return feasible_allocations[i][w];
	}
	/* Returns the transitions between server state w for servers of type i and the other serer states.
	   The j-th element of the vector is the next server state on arrivals of class j. */
	const std::vector<uint32_t> & get_arrival_transitions(uint32_t i, uint32_t w) const {
		return arrival_transitions[i][w];
	}
	/* Returns the transitions between server state w for servers of type i and the other serer states.
	   The j-th element of the vector is the next server state on departures of class j. */
	const std::vector<uint32_t> & get_departure_transitions(uint32_t i, uint32_t w) const {
		return departure_transitions[i][w];
	}
	/* Finds the next system state, starting from state with index z,
	   when the stae of group i changes to next_state_i. */
	uint64_t find_next_system_state(
		const uint64_t z,
		const std::vector<uint64_t> & state,
		const uint32_t i,
		const uint64_t next_state_i) const;
	/* Finds the next group state for group i, starting from the state with index zi,
	   when a server of the group changes state from w to w_next. */
	uint64_t find_next_group_state(
		const uint32_t i,
		const uint64_t zi,
		const std::vector<uint32_t> & group_state,
		const uint32_t w,
		const uint32_t w_next) const;
private:
	/* Used to enumerate the feasible allocations for a server of type i. */
	void enumerate_feasible_allocations_recursive(
		uint32_t i,
		const std::vector<uint32_t> & allocation,
		const std::vector<uint32_t> & used,
		uint32_t level);
	/* Enumerates the feasible allocations for a server of type i. */
	void enumerate_feasible_allocations(uint32_t i);
	/* Enumerates the feasible allocations for each server type. */
	void enumerate_feasible_allocations();
	/* Enumerates all the states for each type of servers. */
	void enumerate_states();
	/* Evaluates which classes of VMs are allowed to arrive in state z. */
	void evaluate_allowed_arrivals(uint64_t z, std::vector<bool> & allowed);
	/* Enumerates all actions. */
	void enumerate_actions();
	/* Lists transitions from one allocation to another on arrivals. */
	void enumerate_arrival_transitions();
	/* Lists transitions from one allocation to another on departures. */
	void enumerate_departure_transitions();
	/* Finds the index of the given allocation for servers of type i. */
	uint32_t find_allocation(uint32_t i, const std::vector<uint32_t> & alloc);
	/* Computes the cost associated with the given state. */
	float compute_state_cost(const state_info & info) const;
	/* Computes the revenue associated with the given state. */
	float compute_state_revenue(const state_info & info) const;
	/* Computes the arrival rate in the given state. */
	float compute_state_arrival_rate(const state_info & info) const;
	/* Computes the departure rate in the given state. */
	float compute_state_departure_rate(const state_info & info) const;
	/* Computes the optimal stationary policy */
	void compute_optimal_policy(bool ignore_revenue, bool allow_unrequired_drop);
public:
	/* Prints a system state. */
	void print_system_state(uint64_t z);
	/* Prints a description of the action at the given index. */
	void print_action(uint32_t a);
private:
	/* Problem instance */
	problem p;
	/* Whether analyze() has been called or not */
	bool analysis;
	/* Total number of states */
	uint64_t state_space_size;
	/* A server of type i can have any allocation in feasible_allocations[i].
	 * An allocation is a vector with the number of VM of each class.
	 * A server of type i in allocation w has x VMs of class j.
	 * x = feasible_allocations[i][w][j];
	 */
	std::vector<std::vector<std::vector<uint32_t>>> feasible_allocations;
	/* A server of type i with allocation w, on an arrival of class j, transitions to allocation w2.
	 * uint32_t w2 = arrival_transitions[i][w][j];
	 * If w2 == w the transition does not exist.
	 */
	std::vector<std::vector<std::vector<uint32_t>>> arrival_transitions;
	/* A server of type i with allocation w, on an departure of class j, transitions to allocation w2.
	 * uint32_t w2 = arrival_transitions[i][w][j];
	 * If w2 == w the transition does not exist.
	 */
	std::vector<std::vector<std::vector<uint32_t>>> departure_transitions;
	/* For each server type, the states for that group of server.
	 * The composite state is obtained by combining the the states of each server type.
	 * For instance, if servers of type 0 are in state 12 out of 35 states and servers of type 1 are in state 8 out of 10 states,
	 * the composite state is 12 + 8 * 35 = 292.*/
	std::vector<comb_with_rep_v2> states;
	/* Vector of all feasible actions. An action specifies where new VM should be provisioned.
	 * A given action a, for each VM class j, specifies two things:
	 * First, the server group i to which arrivals of class j are directed:
	 * uint32_t i = actions[a][j].server_type;
	 * Second, that arrival of class j are directed to a server of group i currently with allocation w.
	 * uint32_t w = actions[a][j].server_alloc;
	 * An action is feasible if in the current state there is at least one server of type i with allocation w.
	 */
	std::vector<std::vector<action>> actions;
};

#endif
