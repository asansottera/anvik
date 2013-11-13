#include "examples.h"

problem example01() {

	uint32_t k = 4; // number of server types
	uint32_t m = 3; // number of VM types
	uint32_t r = 2; // number of resources

	// number of servers of each type
	std::vector<uint32_t> n(k);
	n[0] = 10;
	n[1] = 10;
	n[2] = 5;
	n[3] = 5;

	// capacity of each server type
	std::vector<std::vector<uint32_t>> c(k);
	c[0] = std::vector<uint32_t>(r);
	c[0][0] = 4;
	c[0][1] = 8;
	c[1] = std::vector<uint32_t>(r);
	c[1][0] = 8;
	c[1][1] = 16;
	c[2] = std::vector<uint32_t>(r);
	c[2][0] = 8;
	c[2][1] = 32;
	c[3] = std::vector<uint32_t>(r);
	c[3][0] = 20;
	c[3][1] = 64;

	// requirements
	std::vector<std::vector<uint32_t>> l(m);
	l[0] = std::vector<uint32_t>(r);
	l[0][0] = 2;
	l[0][1] = 4;
	l[1] = std::vector<uint32_t>(r);
	l[1][0] = 2;
	l[1][1] = 8;
	l[2] = std::vector<uint32_t>(r);
	l[2][0] = 8;
	l[2][1] = 16;

	// arrival probabilities
	std::vector<float> lambda(m);
	lambda[0] = 0.05f;
	lambda[1] = 0.03f;
	lambda[2] = 0.02f;

	// departure probabilities
	std::vector<float> mu(m);
	mu[0] = 0.06f;
	mu[1] = 0.04f;
	mu[2] = 0.03f;

	// cost of turning servers on
	std::vector<float> cost(k);
	cost[0] = 1.0;
	cost[1] = 2.0;
	cost[2] = 2.5;
	cost[3] = 4.0;

	// revenue for running VMs
	std::vector<float> revenue(m);
	revenue[0] = 1.0;
	revenue[1] = 3.0;
	revenue[2] = 4.0;

	problem p;
	p.m = m;
	p.k = k;
	p.r = r;
	p.n = n;
	p.c = c;
	p.l = l;
	p.lambda = lambda;
	p.mu = mu;
	p.cost = cost;
	p.revenue = revenue;

	return p;
}

problem example02() {

	uint32_t k = 1; // number of server types
	uint32_t m = 3; // number of VM types
	uint32_t r = 2; // number of resources

	// number of servers of each type
	std::vector<uint32_t> n(k);
	n[0] = 30;

	// capacity of each server type
	std::vector<std::vector<uint32_t>> c(k);
	c[0] = std::vector<uint32_t>(r);
	c[0][0] = 8;
	c[0][1] = 16;

	// requirements
	std::vector<std::vector<uint32_t>> l(m);
	l[0] = std::vector<uint32_t>(r);
	l[0][0] = 2;
	l[0][1] = 4;
	l[1] = std::vector<uint32_t>(r);
	l[1][0] = 2;
	l[1][1] = 8;
	l[2] = std::vector<uint32_t>(r);
	l[2][0] = 8;
	l[2][1] = 16;

	// arrival probabilities
	std::vector<float> lambda(m);
	lambda[0] = 0.05f;
	lambda[1] = 0.03f;
	lambda[2] = 0.02f;

	// departure probabilities
	std::vector<float> mu(m);
	mu[0] = 0.06f;
	mu[1] = 0.04f;
	mu[2] = 0.03f;

	// cost of turning servers on
	std::vector<float> cost(k);
	cost[0] = 1.0;

	// revenue for running VMs
	std::vector<float> revenue(m);
	revenue[0] = 2.0;
	revenue[1] = 3.0;
	revenue[2] = 4.0;

	problem p;
	p.m = m;
	p.k = k;
	p.r = r;
	p.n = n;
	p.c = c;
	p.l = l;
	p.lambda = lambda;
	p.mu = mu;
	p.cost = cost;
	p.revenue = revenue;

	return p;
}

problem example03(uint32_t N) {

	uint32_t k = 1; // number of server types
	uint32_t m = 1; // number of VM types
	uint32_t r = 2; // number of resources

	// number of servers of each type
	std::vector<uint32_t> n(k);
	n[0] = N;

	// capacity of each server type
	std::vector<std::vector<uint32_t>> c(k);
	c[0] = std::vector<uint32_t>(r);
	c[0][0] = 8;
	c[0][1] = 16;

	// requirements
	std::vector<std::vector<uint32_t>> l(m);
	l[0] = std::vector<uint32_t>(r);
	l[0][0] = 2;
	l[0][1] = 4;

	// arrival probabilities
	std::vector<float> lambda(m);
	lambda[0] = 0.05f;

	// departure probabilities
	std::vector<float> mu(m);
	mu[0] = 0.06f;

	// cost of turning servers on
	std::vector<float> cost(k);
	cost[0] = 1.0;

	// revenue for running VMs
	std::vector<float> revenue(m);
	revenue[0] = 1.0;

	problem p;
	p.m = m;
	p.k = k;
	p.r = r;
	p.n = n;
	p.c = c;
	p.l = l;
	p.lambda = lambda;
	p.mu = mu;
	p.cost = cost;
	p.revenue = revenue;

	return p;
}

problem example04() {

	uint32_t k = 1; // number of server types
	uint32_t m = 2; // number of VM types
	uint32_t r = 2; // number of resources

	// number of servers of each type
	std::vector<uint32_t> n(k);
	n[0] = 3;

	// capacity of each server type
	std::vector<std::vector<uint32_t>> c(k);
	c[0] = std::vector<uint32_t>(r);
	c[0][0] = 6;
	c[0][1] = 16;

	// requirements
	std::vector<std::vector<uint32_t>> l(m);
	l[0] = std::vector<uint32_t>(r);
	l[0][0] = 2;
	l[0][1] = 8;
	l[1] = std::vector<uint32_t>(r);
	l[1][0] = 4;
	l[1][1] = 8;

	// arrival probabilities
	std::vector<float> lambda(m);
	lambda[0] = 0.5f;
	lambda[1] = 0.3f;

	// departure probabilities
	std::vector<float> mu(m);
	mu[0] = 0.06f;
	mu[1] = 0.04f;

	// cost of turning servers on
	std::vector<float> cost(k);
	cost[0] = 1.0;

	// revenue for running VMs
	std::vector<float> revenue(m);
	revenue[0] = 1.0;
	revenue[1] = 3.0;

	problem p;
	p.m = m;
	p.k = k;
	p.r = r;
	p.n = n;
	p.c = c;
	p.l = l;
	p.lambda = lambda;
	p.mu = mu;
	p.cost = cost;
	p.revenue = revenue;

	return p;
}

problem example05(uint32_t n1, uint32_t n2) {

	uint32_t k = 2; // number of server types
	uint32_t m = 2; // number of VM types
	uint32_t r = 2; // number of resources

	// number of servers of each type
	std::vector<uint32_t> n(k);
	n[0] = n1;
	n[1] = n2;

	// capacity of each server type
	std::vector<std::vector<uint32_t>> c(k);
	c[0] = std::vector<uint32_t>(r);
	c[0][0] = 6;
	c[0][1] = 16;
	c[1] = std::vector<uint32_t>(r);
	c[1][0] = 4;
	c[1][1] = 16;

	// requirements
	std::vector<std::vector<uint32_t>> l(m);
	l[0] = std::vector<uint32_t>(r);
	l[0][0] = 2;
	l[0][1] = 8;
	l[1] = std::vector<uint32_t>(r);
	l[1][0] = 4;
	l[1][1] = 8;

	// arrival probabilities
	std::vector<float> lambda(m);
	lambda[0] = 0.5;
	lambda[1] = 0.1f;

	// departure probabilities
	std::vector<float> mu(m);
	mu[0] = 0.8f;
	mu[1] = 0.2f;

	// cost of turning servers on
	std::vector<float> cost(k);
	cost[0] = 1.5;
	cost[1] = 1.0;

	// revenue for running VMs
	std::vector<float> revenue(m);
	revenue[0] = 1.0;
	revenue[1] = 3.0;

	problem p;
	p.m = m;
	p.k = k;
	p.r = r;
	p.n = n;
	p.c = c;
	p.l = l;
	p.lambda = lambda;
	p.mu = mu;
	p.cost = cost;
	p.revenue = revenue;

	return p;
}

problem example06(uint32_t n0) {

	uint32_t k = 1; // number of server types
	uint32_t m = 1; // number of VM types
	uint32_t r = 2; // number of resources

	// number of servers of each type
	std::vector<uint32_t> n(k);
	n[0] = n0;

	// capacity of each server type
	std::vector<std::vector<uint32_t>> c(k);
	c[0] = std::vector<uint32_t>(r);
	c[0][0] = 2;
	c[0][1] = 8;

	// requirements
	std::vector<std::vector<uint32_t>> l(m);
	l[0] = std::vector<uint32_t>(r);
	l[0][0] = 2;
	l[0][1] = 8;

	// arrival probabilities
	std::vector<float> lambda(m);
	lambda[0] = 0.5f;

	// departure probabilities
	std::vector<float> mu(m);
	mu[0] = 1.0f;

	// cost of turning servers on
	std::vector<float> cost(k);
	cost[0] = 1.0;

	// revenue for running VMs
	std::vector<float> revenue(m);
	revenue[0] = 2.0;

	problem p;
	p.m = m;
	p.k = k;
	p.r = r;
	p.n = n;
	p.c = c;
	p.l = l;
	p.lambda = lambda;
	p.mu = mu;
	p.cost = cost;
	p.revenue = revenue;

	return p;
}

problem example07(uint32_t n0) {

	uint32_t k = 1; // number of server types
	uint32_t m = 1; // number of VM types
	uint32_t r = 2; // number of resources

	// number of servers of each type
	std::vector<uint32_t> n(k);
	n[0] = n0;

	// capacity of each server type
	std::vector<std::vector<uint32_t>> c(k);
	c[0] = std::vector<uint32_t>(r);
	c[0][0] = 4;
	c[0][1] = 16;

	// requirements
	std::vector<std::vector<uint32_t>> l(m);
	l[0] = std::vector<uint32_t>(r);
	l[0][0] = 2;
	l[0][1] = 8;

	// arrival probabilities
	std::vector<float> lambda(m);
	lambda[0] = 0.5f;

	// departure probabilities
	std::vector<float> mu(m);
	mu[0] = 1.0f;

	// cost of turning servers on
	std::vector<float> cost(k);
	cost[0] = 1.0;

	// revenue for running VMs
	std::vector<float> revenue(m);
	revenue[0] = 2.0;

	problem p;
	p.m = m;
	p.k = k;
	p.r = r;
	p.n = n;
	p.c = c;
	p.l = l;
	p.lambda = lambda;
	p.mu = mu;
	p.cost = cost;
	p.revenue = revenue;

	return p;
}