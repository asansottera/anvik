#include "problem.h"
#include "full_model.h"
#include "cpu_optimizer.h"
#include "gpu_optimizer.h"
#include <iostream>
#include <memory>

problem create_problem(int n1, int n2, float lambda, float p1, float mu, float q1) {
	
	problem p;

	p.k = 2;
	p.m = 2;
	p.r = 2;

	p.n.resize(p.k);
	p.n[0] = n1;
	p.n[1] = n2;

	p.lambda.resize(p.m);
	p.lambda[0] = lambda * p1;
	p.lambda[1] = lambda * (1 - p1);

	p.mu.resize(p.m);
	p.mu[0] = mu * q1;
	p.mu[1] = mu * (1 - q1);

	p.cost.resize(p.k);
	p.cost[0] = 0.5f;
	p.cost[1] = 1.5f;

	p.revenue.resize(p.m);
	p.revenue[0] = 0.5f;
	p.revenue[1] = 1.0f;

	p.c.resize(p.k);
	p.c[0].resize(p.r);
	p.c[0][0] = 8;
	p.c[0][1] = 8;
	p.c[1].resize(p.r);
	p.c[1][0] = 12;
	p.c[1][1] = 16;

	p.l.resize(p.m);
	p.l[0].resize(p.r);
	p.l[0][0] = 4;
	p.l[0][1] = 4;
	p.l[1].resize(p.r);
	p.l[1][0] = 6;
	p.l[1][1] = 8;

	return p;
}

enum OptimizerVersion { OPT_CPU, OPT_GPU };

float run(const problem & p, bool alwaysAllowReject, OptimizerVersion optver = OPT_GPU) {
	
	full_model fm(p);

	std::unique_ptr<optimizer> opt;
    if (optver == OPT_CPU) {
        opt.reset(new cpu_optimizer<float>());
    } else {
        opt.reset(new gpu_optimizer());
    }
	opt->set_always_allow_reject(alwaysAllowReject);
	opt->set_check_strict_convergence(true);
	opt->set_ignore_revenue(false);

	opt->optimize(fm);
	float profit = - opt->get_objective();

	return profit;
}

#define INVALID_ARGUMENTS 1
#define INVALID_COMMAND 2

int main(int argc, char ** argv) {

	if (argc != 2) {
		return INVALID_ARGUMENTS;
	}

	std::string command = argv[1];

	if (command == "server_mix") {
		// we vary the number of servers in the two groups
		uint32_t n = 20;
		std::vector<uint32_t> n1_values(3);
		n1_values[0] = 5;
		n1_values[1] = 10;
		n1_values[2] = 15;
		for (uint32_t t = 0; t < n1_values.size(); ++t) {
			uint32_t n1 = n1_values[t];
			uint32_t n2 = n - n1;
			problem p = create_problem(n1, n2, 6.0f, 0.5f, 4.0f, 0.7f);
			float profit_1 = run(p, false);
			float profit_2 = run(p, true);
			std::cout << "n1 = " << n1 << ", n2 = " << n2 << ": " << profit_1 << ", " << profit_2 << std::endl;
		}
	} else if (command == "vm_mix") {
		// we vary the arrival rate mix
		std::vector<float> p1_values(3);
		p1_values[0] = 0.3f;
		p1_values[1] = 0.5f;
		p1_values[2] = 0.7f;
		for (uint32_t t = 0; t < p1_values.size(); ++t) {
			float p1 = p1_values[t];
			problem p = create_problem(5, 15, 6.0f, p1, 4.0f, 0.7f);
			float profit_1 = run(p, false);
			float profit_2 = run(p, true);
			std::cout << "p1 = " << p1 << ", p2 = " << (1-p1) << ": " << profit_1 << ", " << profit_2 << std::endl;
		}
	} else if (command == "total_rate") {
		// we vary the total arrival rate
		std::vector<float> lambda_values(3);
		lambda_values[0] = 3.0f;
		lambda_values[1] = 6.0f;
		lambda_values[2] = 9.0f;
		for (uint32_t t = 0; t < lambda_values.size(); ++t) {
			float lambda = lambda_values[t];
			problem p = create_problem(5, 15, lambda, 0.5f, 4.0f, 0.7f);
			float profit_1 = run(p, false);
			float profit_2 = run(p, true);
			std::cout << "lambda = " << lambda << ": " << profit_1 << ", " << profit_2 << std::endl;
		}
    } else if (command == "bench") {
        problem p = create_problem(5, 15, 6.0f, 0.7f, 4.0f, 0.7f);
        run(p, true, OPT_CPU);
        run(p, true,  OPT_GPU);
	} else {
		return INVALID_COMMAND;
	}

	return 0;
}
