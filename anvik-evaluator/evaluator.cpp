#include "evaluator.h"

#include <iostream>
#include <cstdint>
#include <chrono>
#include <limits>

#include "cpu_optimizer.h"

// avoids warnings due to the fact that sparse algorithm use 32-bit indices
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE  int
#include <Eigen/Sparse>

void evaluator::evaluate(const full_model & fm, const std::vector<uint32_t> & policy) {
	if (fm.get_system_states() > std::numeric_limits<int>::max()) {
		throw std::runtime_error("Evaluator is limited to systems with 2^31 states");
	}
	const problem & p = fm.get_problem();
	// get number of system states
	int zcount = static_cast<int>(fm.get_system_states());
	// compute tau
	float tau = compute_tau<float>(fm);
	// initialize matrix and right hand side
	Eigen::VectorXf RHS = Eigen::RowVectorXf::Zero(zcount);
	RHS[0] = 1;
	Eigen::SparseMatrix<float> Q(zcount, zcount);
	Q.reserve(Eigen::VectorXi::Constant(zcount, 2 * p.m + 1));
	// total arrival rate
	float total_arrival_rate = 0.0f;
	for (uint32_t j = 0; j < p.m; ++j) {
		total_arrival_rate += p.lambda[j];
	}
	// fill sparse matrix
	full_model::state_info info;
	for (int z = 0; z < zcount; ++z) {
		// get state
		fm.get_system_state_info(z, info);
		// get action
		uint32_t a = policy[z];
		const std::vector<full_model::action> & av = fm.get_action(a);
		// non-zero element on the diagonal
		Q.insert(z,z) = - (total_arrival_rate + info.departure_rate);
		// non-zero elements due to arrivals
		for (uint32_t j = 0; j < p.m; ++j) {
			const full_model::action & aj = av[j];
			if (!aj.drop) {
				// if not dropped, change state
				uint32_t i = aj.server_type;
				uint32_t w = aj.server_alloc;
				uint32_t w_next = fm.get_arrival_transitions(i, w)[j];
				uint64_t zi_next = fm.find_next_group_state(i, info.state[i], info.group_states[i], w, w_next);
				int z_next = static_cast<int>( fm.find_next_system_state(z, info.state, i, zi_next) );
				Q.insert(z_next,z) = p.lambda[j];
			} else {
				Q.coeffRef(z, z) += p.lambda[j];
			}
		}
		// non-zero elements due to departures
		for (uint32_t i = 0; i < p.k; ++i) {
			uint32_t server_states = fm.get_server_states(i);
			for (uint32_t w = 0; w < server_states; ++w) {
				uint32_t servers = info.group_states[i][w];
				if (servers > 0) {
					for (uint32_t j = 0; j < p.m; ++j) {
						uint32_t vms = fm.get_server_state(i, w)[j];
						if (vms > 0) {
							float rate = servers * vms * p.mu[j];
							uint32_t w_next = fm.get_departure_transitions(i, w)[j];
							uint64_t zi_next = fm.find_next_group_state(i, info.state[i], info.group_states[i], w, w_next);
							int z_next = static_cast<int>( fm.find_next_system_state(z, info.state, i, zi_next) );
							Q.insert(z_next, z) = rate;
						}
					}
				}
			}
		}
	}
	// normalize probabilities
	for (int z = 0; z < zcount; ++z) {
		Q.coeffRef(0,z) = 1;
	}
	// compress format
	Q.makeCompressed();
	// solve sparse system
	Eigen::SparseLU<Eigen::SparseMatrix<float>> lu;
	lu.analyzePattern(Q);
	lu.factorize(Q);
	Eigen::VectorXf pi = lu.solve(RHS);
	// compute average cost and average revenue
	cost = 0.0f;
	revenue = 0.0f;
	for (int z = 0; z < zcount; ++z) {
		// get state
		fm.get_system_state_info(z, info);
		// update cost and revenue
		cost += info.cost * pi[z];
		revenue += info.revenue * pi[z];
	}
}