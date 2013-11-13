#include "problem_io.h"
#include "print_utilities.h"
#include "parse_utilities.h"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <functional>
#include <locale>
#include <cctype>

problem_load_error::problem_load_error(const std::string & msg) : std::runtime_error(msg) {
}

problem_save_error::problem_save_error(const std::string & msg) : std::runtime_error(msg) {
}

class problem_invalid_error : public problem_load_error {
public:
	problem_invalid_error(const std::string & msg) : problem_load_error(msg) {
	}
	static problem_invalid_error invalid_line(unsigned count) {
		std::stringstream msg;
		msg << "Invalid problem file. ";
		msg << "Line " << count << " is not a key-value pair separated by ':'.";
		return problem_invalid_error(msg.str());
	}
	static problem_invalid_error invalid_key(unsigned count, const std::string & key) {
		std::stringstream msg;
		msg << "Invalid problem file. ";
		msg << "Line " << count << " has invalid key '" << key << "'";
		return problem_invalid_error(msg.str());
	}
	static problem_invalid_error invalid_value(unsigned count, const std::string & key, const std::string & value) {
		std::stringstream msg;
		msg << "Invalid problem file. ";
		msg << "Line " << count << " has an invalid value for '" << key << "': '" << value << "'";
		return problem_invalid_error(msg.str());
	}
	static problem_invalid_error incomplete(const std::vector<std::string> & keys) {
		std::stringstream msg;
		msg << "Invalid problem file. ";
		msg << "Missing keys: ";
		for (std::size_t i = 0; i < keys.size() - 1; ++i) {
			msg << keys[i] << ", ";
		}
		msg << keys[keys.size()-1];
		return problem_invalid_error(msg.str());
	}
	static problem_invalid_error invalid_vector_length(const std::string & key) {
		std::stringstream msg;
		msg << "Invalid problem file. ";
		msg << "Vector value for '" << key << "' has an invalid length";
		return problem_invalid_error(msg.str());
	}
	static problem_invalid_error duplicate_key(unsigned count, const std::string & key) {
		std::stringstream msg;
		msg << "Invalid problem file. ";
		msg << "Duplicates for key '" << key << "' at line " << count;
		return problem_invalid_error(msg.str());
	}
};

problem load_problem(const std::string & fname) {
	problem p;

	std::ifstream f(fname);

	if (f.fail()) {
		throw problem_load_error("Unable to open problem file for input");
	}

	unsigned count = 0;
	std::string line;
	std::vector<std::string> key_value;

	bool found_k = false;
	bool found_m = false;
	bool found_r = false;
	bool found_n = false;
	bool found_lambda = false;
	bool found_mu = false;
	bool found_cost = false;
	bool found_revenue = false;
	bool found_capacity = false;
	bool found_requirement = false;
	bool found_r_names = false;
	bool found_s_names = false;
	bool found_v_names = false;

	while (!f.eof()) {
		++count;
		std::getline(f, line);
		// trim spaces
		trim(line);
		// ignore empty lines
		if (line.size() == 0) {
			continue;
		}
		split(line, ':', key_value);
		if (key_value.size() != 2) {
			throw problem_invalid_error::invalid_line(count);
		}
		std::string & key = key_value[0];
		std::string & value = key_value[1];
		trim(key);
		trim(value);
		if (key == "k") {
			if (!found_k) {
				if (!try_parse(value, p.k)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_k = true;
		} else if (key == "m") {
			if (!found_m) {
				if (!try_parse(value, p.m)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_m = true;
		} else if (key == "r") {
			if (!found_r) {
				if (!try_parse(value, p.r)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_r = true;
		} else if (key == "n") {
			if (!found_n) {
				if (!try_parse(value, p.n)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_n = true;
		} else if (key == "lambda") {
			if (!found_lambda) {
				if (!try_parse(value, p.lambda)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_lambda = true;
		} else if (key == "mu") {
			if (!found_mu) {
				if (!try_parse(value, p.mu)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_mu = true;
		} else if (key == "cost") {
			if (!found_cost) {
				if (!try_parse(value, p.cost)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_cost = true;
		} else if (key == "revenue") {
			if (!found_revenue) {
				if (!try_parse(value, p.revenue)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_revenue = true;
		} else if (key == "capacity") {
			if (!found_capacity) {
				if (!try_parse(value, p.c)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_capacity = true;
		} else if (key == "requirement") {
			if (!found_requirement) {
				if (!try_parse(value, p.l)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_requirement = true;
		} else if (key == "r_names") {
			if (!found_r_names) {
				if (!try_parse(value, p.r_names)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_r_names = true;
		} else if (key == "s_names") {
			if (!found_s_names) {
				if (!try_parse(value, p.s_names)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_s_names = true;
		} else if (key == "v_names") {
			if (!found_v_names) {
				if (!try_parse(value, p.v_names)) {
					throw problem_invalid_error::invalid_value(count, key, value);
				}
			} else {
				throw problem_invalid_error::duplicate_key(count, key);
			}
			found_v_names = true;
		} else {
			throw problem_invalid_error::invalid_key(count, key);
		}
	}

	std::vector<std::string> missing;
	if (!found_k) missing.push_back("k");
	if (!found_m) missing.push_back("m");
	if (!found_r) missing.push_back("r");
	if (!found_n) missing.push_back("n");
	if (!found_lambda) missing.push_back("lambda");
	if (!found_mu) missing.push_back("mu");
	if (!found_cost) missing.push_back("cost");
	if (!found_revenue) missing.push_back("revenue");
	if (!found_capacity) missing.push_back("capacity");
	if (!found_requirement) missing.push_back("requirement");
	if (missing.size() > 0) {
		throw problem_invalid_error::incomplete(missing);
	}

	p.check();

	return p;
}

void save_problem(const std::string & fname, const problem & p) {

	std::ofstream f(fname);

	if (f.fail()) {
		throw problem_save_error("Unable to open problem file for output");
	}

	f << "k: " << p.k << std::endl;
	f << "m: " << p.m << std::endl;
	f << "r: " << p.r << std::endl;
	f << "n: " << p.n << std::endl;
	f << "lambda: " << p.lambda << std::endl;
	f << "mu: " << p.mu << std::endl;
	f << "cost: " << p.cost << std::endl;
	f << "revenue: " << p.revenue << std::endl;
	f << "capacity: " << p.c << std::endl;
	f << "requirement: " << p.l << std::endl;
	if (p.r_names.size() > 0) f << "r_names: " << p.r_names << std::endl;
	if (p.s_names.size() > 0) f << "s_names: " << p.s_names << std::endl;
	if (p.v_names.size() > 0) f << "v_names: " << p.v_names << std::endl;

	f.close();

}

policy_load_error::policy_load_error(const std::string & msg) : std::runtime_error(msg) {
}

policy_save_error::policy_save_error(const std::string & msg) : std::runtime_error(msg) {
}

std::vector<uint32_t> load_policy(const std::string & policy_file) {
	std::ifstream input(policy_file);
	if (input.fail()) {
		throw policy_load_error("Unable to open policy file for reading");
	}
	std::vector<uint32_t> policy;
	if (!try_parse(input, policy)) {
		throw policy_load_error("Invalid policy file");
	}
	return policy;
}

void save_policy(const std::string & policy_file, const std::vector<uint32_t> & policy) {
	std::ofstream output(policy_file);
	if (output.fail()) {
		throw policy_save_error("Unable to open policy file for writing");
	}
	output << policy << std::endl;
	output.close();
}