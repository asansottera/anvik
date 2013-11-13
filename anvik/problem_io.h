#ifndef ANVIK_PROBLEM_IO_H
#define ANVIK_PROBLEM_IO_H

#include <string>
#include <stdexcept>
#include <vector>
#include <cstdint>

#include "problem.h"

class problem_load_error : public std::runtime_error {
public:
	problem_load_error(const std::string & msg);
};

class problem_save_error : public std::runtime_error {
public:
	problem_save_error(const std::string & msg);
};

problem load_problem(const std::string & fname);
void save_problem(const std::string & fname, const problem & p);

class policy_load_error : public std::runtime_error {
public:
	policy_load_error(const std::string & msg);
};

class policy_save_error : public std::runtime_error {
public:
	policy_save_error(const std::string & msg);
};

std::vector<uint32_t> load_policy(const std::string & policy_file);
void save_policy(const std::string & policy_file, const std::vector<uint32_t> & policy);

#endif