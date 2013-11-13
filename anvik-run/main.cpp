#include <cstdint>
#include <vector>
#include <limits>
#include <iostream>
#include <utility>
#include <sstream>
#include <memory>
#include <fstream>
#include <chrono>

#include "problem.h"
#include "full_model.h"
#include "cpu_optimizer.h"
#include "gpu_optimizer.h"
#include "problem_io.h"
#include "parse_utilities.h"
#include "print_utilities.h"
#include "simulator.h"
#include "evaluator.h"

void optimize(const std::string & problem_file, const std::string & policy_file, bool use_gpu = true) {
	// load problem
	problem p = load_problem(problem_file);
	// write problem to command line
	std::cout << p;
	// analyze problem
	full_model fm;
	fm.analyze(p);
	// optimize problem
	std::unique_ptr<optimizer> opt = 0;
	if (!use_gpu) {
		std::cout << "Optimizing on CPU" << std::endl;
		opt.reset( new cpu_optimizer<float>() );
	} else {
		std::cout << "Optimizing on GPU" << std::endl;
		opt.reset( new gpu_optimizer() );
	}
	opt->set_ignore_revenue(false);
	opt->set_always_allow_reject(false);
	opt->set_check_strict_convergence(true);
	opt->optimize(fm);
	// save policy
	save_policy(policy_file, opt->get_policy());
}

void evaluate(const std::string & problem_file, const std::string & policy_file) {
	problem p = load_problem(problem_file);
	std::vector<uint32_t> policy = load_policy(policy_file);
	full_model fm(p);
	evaluator evaluator;
	std::cout << "Evaluating policy..." << std::endl;
	std::chrono::high_resolution_clock::time_point start, end;
	start = std::chrono::high_resolution_clock::now();
	evaluator.evaluate(fm, policy);
	end = std::chrono::high_resolution_clock::now();
	long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
	std::cout << "Evaluator completed in " << ms << " ms" << std::endl;
	std::cout << "Evaluator cost: " << evaluator.get_average_cost() << std::endl;
	std::cout << "Evaluator revenue: " << evaluator.get_average_revenue() << std::endl;
	std::cout << "Evaluator lost: " << evaluator.get_average_loss() << std::endl;
}

void simulate(const std::string & problem_file, const std::string & policy_file) {
	problem p = load_problem(problem_file);
	std::vector<uint32_t> policy = load_policy(policy_file);
	full_model fm(p);
	simulator sim;
	std::cout << "Simulating policy..." << std::endl;
	std::chrono::high_resolution_clock::time_point start, end;
	start = std::chrono::high_resolution_clock::now();
	sim.simulate(fm, policy);
	end = std::chrono::high_resolution_clock::now();
	long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
	std::cout << "Simulator completed in " << ms << " ms" << std::endl;
	std::cout << "Simulator cost: " << sim.get_average_cost() << std::endl;
	std::cout << "Simulator revenue: " << sim.get_average_revenue() << std::endl;
	std::cout << "Simulator lost: " << sim.get_average_loss() << std::endl;
}

#ifdef _WIN32
#include <Windows.h>
#endif

#define ANVIK_RUN_INVALID_NUMBER_OF_ARGUMENTS 1
#define ANVIK_RUN_INVALID_COMMAND 2
#define ANVIK_RUN_ERROR 3

int main(int argc, char ** argv) {

	if (argc != 4) {
		std::cerr << "Invalid number of arguments" << std::endl;
		return ANVIK_RUN_INVALID_NUMBER_OF_ARGUMENTS;
	}

	std::string cmd = argv[1];
	std::string problem_file = argv[2];
	std::string policy_file = argv[3];

	try {

		if (cmd == "optimize-cpu") {
			optimize(problem_file, policy_file, false);
		} else if (cmd == "optimize-gpu") {
			optimize(problem_file, policy_file, true);
		} else if (cmd == "evaluate") {
			evaluate(problem_file, policy_file);
		} else if (cmd == "simulate") {
			simulate(problem_file, policy_file);
		} else {
			std::cerr << "Invalid commmand '" << cmd << "'" << std::endl;
			return ANVIK_RUN_INVALID_COMMAND;
		}

		return 0;

	} catch (std::runtime_error & err) {
		std::cerr << err.what() << std::endl;
		return ANVIK_RUN_ERROR;
	}
}
