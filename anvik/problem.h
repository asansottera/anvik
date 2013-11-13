#ifndef ANVIK_PROBLEM_H
#define ANVIK_PROBLEM_H

#include <cstdint>
#include <vector>
#include <ostream>
#include <string>

struct problem {
	/* Number of server groups. */
	uint32_t k;
	/* Number of VM classes. */
	uint32_t m;
	/* Number of resources. */
	uint32_t r;
	/* Number of servers in each group. */
	std::vector<uint32_t> n;
	/* For each server group, the r-vector of capacity. */
	std::vector<std::vector<uint32_t>> c;
	/* For each VM class, the r-vector of requirements. */
	std::vector<std::vector<uint32_t>> l;
	/* Service rate for each VM class. */
	std::vector<float> mu;
	/* Arrival rate for each VM class. */
	std::vector<float> lambda;
	/* For each server group, the cost rate of a server. */
	std::vector<float> cost;
	/* For each VM class, the revenue rate of a server. */
	std::vector<float> revenue;
	/* Names of the resources (optional, used by UI). */
	std::vector<std::string> r_names;
	/* Names of the server types (optional, used by UI). */
	std::vector<std::string> s_names;
	/* Names of the VM classes (optional, used by UI). */
	std::vector<std::string> v_names;
	/* Checks the consistency of the problem. */
	void check() const;
};

std::ostream & operator<<(std::ostream & out, const problem & p);

#endif