#include "problem.h"
#include "print_utilities.h"

#include <stdexcept>
#include <sstream>

void problem::check() const {
	if (k != n.size()) {
		throw std::runtime_error("Vector with number of servers has wrong size");
	}
	if (k != cost.size()) {
		throw std::runtime_error("Cost vector has wrong size");
	}
	if (c.size() != k) {
		throw std::runtime_error("Capacity vector of vectors has wrong size");
	}
	for (uint32_t i = 0; i < k; ++i) {
		if (r != c[i].size()) {
			throw std::runtime_error("Capacity vector has wrong size");
		}
	}
	if (l.size() != m) {
		throw std::runtime_error("Requirement vector of vectors has wrong size");
	}
	for (uint32_t j = 0; j < m; ++j) {
		if (r != l[j].size()) {
			throw std::runtime_error("Requirement vector has wrong size");
		}
	}
	if (m != lambda.size()) {
		throw std::runtime_error("Arrival rate vector has wrong size");
	}
	if (m != mu.size()) {
		throw std::runtime_error("Departure rate vector has wrong size");
	}
	if (m != revenue.size()) {
		throw std::runtime_error("Revenue vector has wrong size");
	}
	if (!r_names.empty() && r_names.size() != r) {
		throw std::runtime_error("Vector of resource names has wrong size");
	}
	if (!r_names.empty()) {
		for (uint32_t h = 0; h < r; ++h) {
			if (r_names[h].find_first_of(",[]{}") != std::string::npos) {
				std::stringstream msg;
				msg << "Resource name '" << r_names[h] << "' contains a reserved character";
				throw std::runtime_error(msg.str());
			}
		}
	}
	if (!s_names.empty() && s_names.size() != k) {
		throw std::runtime_error("Vector of server group names has wrong size");
	}
	if (!s_names.empty()) {
		for (uint32_t i = 0; i < k; ++i) {
			if (s_names[i].find_first_of(",[]{}") != std::string::npos) {
				std::stringstream msg;
				msg << "Serer group name '" << s_names[i] << "' contains a reserved character";
				throw std::runtime_error(msg.str());
			}
		}
	}
	if (!v_names.empty() && v_names.size() != m) {
		throw std::runtime_error("Vector of virtual machine class names has wrong size");
	}
	if (!v_names.empty()) {
		for (uint32_t j = 0; j < m; ++j) {
			if (v_names[j].find_first_of(",[]{}") != std::string::npos) {
				std::stringstream msg;
				msg << "Virtual machine class name '" << v_names[j] << "' contains a reserved character";
				throw std::runtime_error(msg.str());
			}
		}
	}
}

std::ostream & operator<<(std::ostream & out, const problem & p) {
	out << "Number of servers of each type: " << p.n << std::endl;
	for (uint32_t i = 0; i < p.k; ++i) {
		out << "Capacities of servers of type " << i << ": " << p.c[i] << std::endl;
	}
	for (uint32_t j = 0; j < p.m; ++j) {
		out << "Requirements of VMs of type " << j << ": " << p.l[j] << std::endl;
	}
	out << "Arrival rates: " << p.lambda << std::endl;
	out << "Departure rates: " << p.mu << std::endl;
	out << "Server costs: " << p.cost << std::endl;
	out << "VM revenues: " << p.revenue << std::endl;
	return out;
}