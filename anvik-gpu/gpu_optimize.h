#ifndef ANVIK_GPU_KERNEL_H
#define ANVIK_GPU_KERNEL_H

#include "gpu_problem.h"
#include "gpu_analysis.h"
#include "gpu_optdata.h"

bool gpu_check_requirements();

void gpu_optimize_init(const gpu_problem & gp , const gpu_analysis & ga, gpu_optdata & go);

void gpu_optimize_iteration(bool ignoreRevenue, bool alwaysAllowReject, bool check_strict_convergence,
							unsigned iteration, float & cost, float & delta);

void gpu_get_policy(std::vector<uint32_t> & policy);

#endif