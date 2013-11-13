#ifndef ANVIK_OPTIMIZER_H
#define ANVIK_OPTIMIZER_H

#include <vector>
#include "full_model.h"

struct worker;

class optimizer {
public:
	optimizer() {
		ignore_revenue = true;
		always_allow_reject = false;
		check_strict_convergence = false;
	}
	virtual ~optimizer() { }
	virtual void optimize(const full_model & fm) = 0;
	virtual void start_optimize(const full_model & fm) = 0;
	virtual void cancel_optimize() = 0;
	virtual void join_optimize() = 0;
public:
	bool get_ignore_revenue() const {
		return ignore_revenue;
	}
	void set_ignore_revenue(bool value) {
		ignore_revenue = value;
	}
	bool get_always_allow_reject() const {
		return always_allow_reject;
	}
	void set_always_allow_reject(bool value) {
		always_allow_reject = value;
	}
	bool get_check_strict_convergence() const {
		return check_strict_convergence;
	}
	void set_check_strict_convergence(bool value) {
		check_strict_convergence = value;
	}
	virtual bool is_available() const = 0;
	virtual float get_objective() const = 0;
	virtual const std::vector<uint32_t> & get_policy() const = 0;
	virtual uint32_t get_iteration() const = 0;
protected:
	bool ignore_revenue;
	bool always_allow_reject;
	bool check_strict_convergence;
};

#endif