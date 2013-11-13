#ifndef ANVIK_COMBINATORICS_H
#define ANVIK_COMBINATORICS_H

#include <limits>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include "matrix.h"

inline double factorial(uint32_t a) {
	double res = 1;
	for (uint32_t i = 1; i <= a; ++i) {
		res *= i;
	}
	return res;
}

inline double binomial(uint32_t a, uint32_t b) {
	double res = factorial(a) / (factorial(b)*factorial(a-b));
	return std::floor(res + 0.5);
}

class choose {
public:
	choose() : N(0), K(0) {
	}
	choose(uint32_t _N, uint32_t _K): N(_N), K(_K) {
		C.resize(N+1, K+1);
		compute();
	}
	void init(uint32_t _N, uint32_t _K) {
		N = _N;
		K = _K;
		C.resize(N+1,K+1);
		compute();
	}
	uint64_t operator()(uint32_t n, uint32_t k) const {
		assert(n <= N && k <= K);
		return C(n,k);
	}
private:
	uint32_t N;
	uint32_t K;
	matrix<uint64_t> C;
	void compute() {
		for (uint64_t k = 1; k <= K; k++) C(0,k) = 0;
		for (uint64_t n = 0; n <= N; n++) C(n,0) = 1;
		for (uint64_t n = 1; n <= N; n++) {
			for (uint64_t k = 1; k <= K; k++) {
				if (std::numeric_limits<uint64_t>::max() - C(n-1,k-1) < C(n-1,k)) {
					std::stringstream msg;
					msg << "Choose overflow: C(" << n << "," << k << ")";
					throw std::runtime_error(msg.str());
				}
				C(n,k) = C(n-1,k-1) + C(n-1,k);
			}
		}
	}
};

class multichoose {
public:
	multichoose(): N(0), K(0) {
	}
	multichoose(uint32_t _N, uint32_t _K): N(_N), K(_K) {
		compute(N, K, C);
	}
	void init(uint32_t _N, uint32_t _K) {
		N = _N;
		K = _K;
		compute(N, K, C);
	}
	uint64_t operator()(uint32_t n, uint32_t k) const {
		assert(k >= 0 && n <= N && k <= K);
		return C(n,k);
	}
private:
	uint32_t N;
	uint32_t K;
	matrix<uint64_t> C;
public:
	static void compute(uint32_t N, uint32_t K, matrix<uint64_t> & C) {
		if (C.rows() != N+1 && C.cols() != K+1) {
			C.resize(N+1,K+1);
		}
		for (uint64_t k = 0; k <= K; k++) C(0,k) = 1;
		for (uint64_t k = 1; k <= K; k++) C(1,k) = k;
		for (uint64_t n = 0; n <= N; n++) C(n,1) = 1;
		for (uint64_t n = 2; n <= N; n++) {
			for (uint64_t k = 2; k <= K; k++) {
				if (std::numeric_limits<uint64_t>::max() - C(n,k-1) < C(n-1,k)) {
					std::stringstream msg;
					msg << "Multichoose overflow: C(" << n << "," << k << ")";
					throw std::runtime_error(msg.str());
				}
				uint64_t tmp = C(n,k-1) + C(n-1,k);
				C(n,k) = tmp;
			}
		}
	}
};


#endif
