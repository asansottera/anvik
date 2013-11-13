#ifndef ANVIK_EVALUATOR_H
#define ANVIK_EVALUATOR_H

#include "problem.h"
#include "full_model.h"
#include <vector>
#include <cstdint>

class evaluator {
public:
	void evaluate(const full_model & fm, const std::vector<uint32_t> & policy);
	float get_average_cost() const {
		return cost;
	}
	float get_average_revenue() const {
		return revenue;
	}
	float get_average_loss() const {
		return (cost - revenue);
	}
	float get_average_profit() const {
		return (revenue - cost);
	}
private:
	float cost;
	float revenue;
};

#endif