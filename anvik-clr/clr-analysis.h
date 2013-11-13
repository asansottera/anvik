#pragma once

#include "clr-problem.h"
#include "full_model.h"
#include "optimizer.h"

using namespace System;
using namespace System::Collections::Generic;

namespace anvik {

	public ref class TooManyStatesException : public Exception
	{
	public:
		TooManyStatesException(String^ msg) : Exception(msg) {
		}
	};

	public ref class OptimizerVersionNotAvailable : public Exception
	{
	public:
		OptimizerVersionNotAvailable() : Exception("The optimizer is not available on this computer") {
		}
	};

	public ref class PolicySaveException : public Exception {
	public:
		PolicySaveException(String^ msg) : Exception(msg) {
		}
	};

	public enum class OptimizerState {
		Ready, Started, Completed
	};

	public enum class OptimizerVersion {
		CpuOptimizer, GpuOptimizer
	};

	public ref class Analysis
	{
		full_model * fm;
		optimizer * opt;
		OptimizerState optState;
	public:
		Analysis(Problem^ p);
		Analysis(const problem & np);
		~Analysis();
		uint64_t GetServerStates(uint32_t i);
		uint64_t GetGroupStates(uint32_t i);
		uint64_t GetSystemStates();
		uint64_t GetActions();
		void StartOptimize(OptimizerVersion optVer, bool ignore_revenue, bool always_allow_reject, bool check_strict_convergence);
		void CancelOptimize();
		void JoinOptimize();
		OptimizerState GetOptimizerState();
		double GetBestObjective();
		uint32_t GetBestPolicy(uint64_t stateIdx);
		void SaveBestPolicy(String^ fname);
		uint32_t GetIteration();
		String^ DescribeState(uint64_t z);
		String^ DescribeAction(uint32_t a);
		uint64_t ComputeDestinationOnArrival(uint64_t z, uint32_t a, uint32_t j);
	};

}