#ifndef ANVIK_CPU_OPTIMIZER_H
#define ANVIK_CPU_OPTIMIZER_H

#include <vector>
#include <limits>
#include "full_model.h"
#include "optimizer.h"

struct worker;

template<class TFloat = float>
class cpu_optimizer : public optimizer {
public:
	cpu_optimizer();
	~cpu_optimizer();
	void optimize(const full_model & fm);
	void start_optimize(const full_model & fm);
	void cancel_optimize();
	void join_optimize();
public:
	float get_objective() const {
		return static_cast<float>(objective);
	}
	const std::vector<uint32_t> & get_policy() const {
		return policy;
	}
	bool is_available() const;
	uint32_t get_iteration() const;
private:
	worker * w;
	/* The optimal stationary policy.
	 * Let the system be in state z, with the server groups in states <y[1],y[2],...y[k]>.
	 * A policy specify the action a to take in state z.
	 * If no arrival or departure occurs, the state does not change, regardless of the policy.
	 * If a departure of class j occurs, the next state does not depend on the policy.
	 * If an arrival of class j occurs, the next state depends on the policy.
	 * In fact, if the policy dictates to route arrivals of class to a server of type i in allocation w,
	 * one server of type i changes its allocation to w2.
	 * uint32_t w2 = arrival_transitions[i][w][j]
	 * The next state z2 is a state <y2[1],y2[2],...,y2[k]> such that:
	 * - for any i2 != i, y2[i2] = y[i2]
	 * - y2[i] differs by y[i] only by two elements
	 *   - servers in allocation w are decreased by 1
	 *   - servers in allocation w2 are increased by 1
	 * The value of y2[i] can be found using find_state(i, state_i) and then z2 can be found using get_state(<y2[1],y2[2],...,y2[k]>).
	 */
	std::vector<uint32_t> policy;
	std::vector<TFloat> via_w_1;
	std::vector<TFloat> via_w_2;
	/* The objective achieved by the optimal stationary policy */
	TFloat objective;
private:
	void optimize_state_iteration(
		const full_model & fm,
		const uint64_t z,
		full_model::state_info & info,
		std::vector<matrix<uint64_t>> & arrival_next_state,
		const TFloat total_arrival_rate,
		const TFloat tau,
		const unsigned iteration);
};

/* Computes the time constant for uniformization. */
template<class TFloat>
TFloat compute_tau(const full_model & fm);

#endif