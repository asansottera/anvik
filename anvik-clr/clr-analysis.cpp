#include "stdafx.h"

#define NOMINMAX
#include <msclr/marshal_cppstd.h>

#include "clr-analysis.h"
#include "cpu_optimizer.h"
#include "gpu_optimizer.h"
#include "problem_io.h"

using namespace anvik;
using namespace System::Text;

Analysis::Analysis(Problem^ p) {
	fm = new full_model();
	opt = 0;
	optState = OptimizerState::Ready;
	problem np = p->ToNative();
	try {
		fm->analyze(np);
	} catch (std::runtime_error & err) {
		throw gcnew TooManyStatesException(gcnew String(err.what()));
	}
}

Analysis::Analysis(const problem & np) {
	fm = new full_model();
	opt = 0;
	optState = OptimizerState::Ready;
	fm->analyze(np);
}

Analysis::~Analysis() {
	delete fm;
	if (opt != 0) delete opt;
}

uint64_t Analysis::GetServerStates(uint32_t i) {
	return fm->get_server_states(i);
}

uint64_t Analysis::GetGroupStates(uint32_t i) {
	return fm->get_group_states(i);
}

uint64_t Analysis::GetSystemStates() {
	return fm->get_system_states();
}

uint64_t Analysis::GetActions() {
	return fm->get_actions();
}

void Analysis::StartOptimize(OptimizerVersion optVer, bool ignore_revenue, bool always_allow_reject, bool check_strict_convergence) {
	if (opt != 0) delete opt;
	if (optVer == OptimizerVersion::CpuOptimizer) {
		opt = new cpu_optimizer<float>;
	} else if (optVer == OptimizerVersion::GpuOptimizer) {
		opt = new gpu_optimizer;
	}
	if (!opt->is_available()) {
		throw gcnew OptimizerVersionNotAvailable();
	}
	opt->set_ignore_revenue(ignore_revenue);
	opt->set_always_allow_reject(always_allow_reject);
	opt->set_check_strict_convergence(check_strict_convergence);
	opt->start_optimize(*fm);
	optState = OptimizerState::Started;
}

void Analysis::CancelOptimize() {
	opt->cancel_optimize();
	optState = OptimizerState::Ready;
}

void Analysis::JoinOptimize() {
	opt->join_optimize();
	optState = OptimizerState::Completed;
}

double Analysis::GetBestObjective() {
	return opt->get_objective();
}

uint32_t Analysis::GetBestPolicy(uint64_t stateIdx) {
	const std::vector<uint32_t> & npolicy = opt->get_policy();
	return npolicy[stateIdx];
}

void Analysis::SaveBestPolicy(String^ fname) {
	std::string fn = msclr::interop::marshal_as<std::string>(fname);
	try {
		save_policy(fn, opt->get_policy());
	} catch (policy_save_error & e) {
		throw gcnew PolicySaveException(gcnew String(e.what()));
	}
}

uint32_t Analysis::GetIteration() {
	return opt->get_iteration();
}

OptimizerState Analysis::GetOptimizerState() {
	return optState;
}

String^ Analysis::DescribeState(uint64_t z) {
	const problem & p = fm->get_problem();
	StringBuilder builder;
	builder.AppendFormat("State {0}", z);
	builder.AppendLine();
	std::vector<uint64_t> state(p.k);
	fm->get_system_state(z, state);
	for (uint32_t i = 0; i < p.k; ++i) {
		builder.AppendFormat("  State for servers of group {0}: {1}", i, state[i]);
		builder.AppendLine();
		uint64_t server_states = fm->get_server_states(i);
		std::vector<uint32_t> group_state(server_states);
		fm->get_group_state(i, state[i], group_state);
		for (uint32_t w = 0; w < server_states; ++w) {
			const std::vector<uint32_t> & server_state = fm->get_server_state(i, w);
			builder.AppendFormat("    {0} servers in state {1}: [", group_state[w], w);
			for (uint32_t j = 0; j < p.m-1; ++j) {
				builder.AppendFormat("{0}, ", server_state[j]);
			}
			builder.AppendFormat("{0}]", server_state[p.m-1]);
			builder.AppendLine();
		}
	}
	return builder.ToString();
}

String^ Analysis::DescribeAction(uint32_t a) {
	const problem & p = fm->get_problem();
	StringBuilder builder;
	builder.AppendFormat("Action {0}", a);
	builder.AppendLine();
	const std::vector<full_model::action> & av = fm->get_action(a);
	for (uint32_t j = 0; j < p.m; ++j) {
		builder.AppendFormat("  VM of class {0}: ", j);
		if (av[j].drop) {
			builder.AppendFormat("drop");
		} else {
			builder.AppendFormat("server of group {0} in state {1}",
				av[j].server_type, 
				av[j].server_alloc);
		}
		builder.AppendLine();
	}
	return builder.ToString();
}

uint64_t Analysis::ComputeDestinationOnArrival(uint64_t z, uint32_t a, uint32_t j) {
	const problem & p = fm->get_problem();
	// get action
	const std::vector<full_model::action> & av = fm->get_action(a);
	if (av[j].drop) {
		return z;
	} else {
		// get state
		std::vector<uint64_t> state(p.k);
		std::vector<uint64_t> state_next(p.k);
		fm->get_system_state(z, state);
		// get action information
		uint32_t i = av[j].server_type;
		uint32_t w = av[j].server_alloc;
		uint32_t w_next = fm->get_arrival_transitions(i,w)[j];
		// compute destionation
		if (w != w_next) {
			// get group state
			uint64_t server_states = fm->get_server_states(i);
			std::vector<uint32_t> group_state(server_states);
			std::vector<uint32_t> group_state_next(server_states);
			fm->get_group_state(i, state[i], group_state);
			// compute state index for servers of type i
			group_state_next = group_state;
			group_state_next[w] -= 1;
			group_state_next[w_next] += 1;
			// compute state index for the whole system
			uint64_t istate_next_idx = fm->find_group_state(i, group_state_next);
			state_next = state;
			state_next[i] = istate_next_idx;
			uint64_t z_next = fm->find_system_state(state_next);
			return z_next;
		} else {
			return z;
		}
	}
}