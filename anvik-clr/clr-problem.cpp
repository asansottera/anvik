// This is the main DLL file.

#include "stdafx.h"

#include <msclr/marshal_cppstd.h>

#include "clr-problem.h"
#include "problem_io.h"

#include <string>

using namespace anvik;

bool Problem::Check(List<String^>^ messages)
{
    if (m_resources->Count == 0)
    {
        messages->Add("At least one resource should be defined");
    }
    if (m_serverGroups->Count == 0)
    {
        messages->Add("At least one server group should be defined");
    }
    if (m_vmClasses->Count == 0)
    {
        messages->Add("At least one virtual machine class should be defined");
    }
    for each (ServerGroup^ sg in m_serverGroups)
    {
        if (sg->Count <= 0)
        {
            messages->Add("Server group \"" + sg->Name + "\" should have at least one server");
        }
        if (sg->Cost < 0)
        {
            messages->Add("Cost for server group \"" + sg->Name + "\" should be non-negative");
        }
        if (sg->Capacity->Count != m_resources->Count)
        {
            messages->Add("Capacity of server group \"" + sg->Name + "\" has inconsistent length");
        }
        bool zero_size = true;
        for (int r = 0; r < sg->Capacity->Count; ++r)
        {
            if (sg->Capacity[r] < 0)
            {
                messages->Add("Capacity of resource \"" + m_resources[r] + "\" for servers of type \"" + sg->Name + "\" should be non-negative");
            }
            if (sg->Capacity[r] > 0)
            {
                zero_size = false;
            }
        }
        if (zero_size)
        {
            messages->Add("Server group \"" + sg->Name + "\" should have a strictly positive capacity for at lest one resource");
        }
    }
    for each (VmClass^ vmc in m_vmClasses)
    {
        if (vmc->ArrivalRate <= 0)
        {
            messages->Add("VM class \"" + vmc->Name + "\" should have a positive arrival rate");
        }
        if (vmc->ServiceRate <= 0)
        {
            messages->Add("VM class \"" + vmc->Name + "\" should have a positive service rate");
        }
        if (vmc->Revenue < 0)
        {
            messages->Add("Revene for VM class \"" + vmc->Name + "\" should be non-negative");
        }
        if (vmc->Requirement->Count != m_resources->Count)
        {
            messages->Add("Requirement of VM class \"" + vmc->Name + "\" has inconsistent length");
        }
        bool zero_size = true;
        for (int r = 0; r < vmc->Requirement->Count; ++r)
        {
            if (vmc->Requirement[r] < 0)
            {
                messages->Add("Requirement of resource \"" + m_resources[r] + "\" for VM class \"" + vmc->Name + "\" should be non-negative");
            }
            if (vmc->Requirement[r] > 0)
            {
                zero_size = false;
            }
        }
        if (zero_size)
        {
            messages->Add("VM class\"" + vmc->Name + "\" should have a strictly positive requirement for at lest one resource");
        }
    }
    return messages->Count == 0;
}

problem Problem::ToNative() {
	problem p;
	p.k = m_serverGroups->Count;
	p.m = m_vmClasses->Count;
	p.r = m_resources->Count;
	p.n.resize(p.k);
	p.c.resize(p.k);
	p.cost.resize(p.k);
	p.r_names.resize(p.r);
	p.s_names.resize(p.k);
	p.v_names.resize(p.m);
	for (uint32_t h = 0; h < p.r; ++h) {
		String^ rname = m_resources[h];
		p.r_names[h] = msclr::interop::marshal_as<std::string>(rname);
	}
	for (uint32_t i = 0; i < p.k; ++i) {
		ServerGroup^ sg = m_serverGroups[i];
		p.s_names[i] = msclr::interop::marshal_as<std::string>(sg->Name);
		p.n[i] = sg->Count;
		p.c[i].resize(p.r);
		for (uint32_t h = 0; h < p.r; ++h) {
			p.c[i][h] = sg->Capacity[h];
		}
		p.cost[i] = sg->Cost;
	}
	p.l.resize(p.m);
	p.lambda.resize(p.m);
	p.mu.resize(p.m);
	p.revenue.resize(p.m);
	for (uint32_t j = 0; j < p.m; ++j) {
		VmClass^ vmc = m_vmClasses[j];
		p.v_names[j] = msclr::interop::marshal_as<std::string>(vmc->Name);
		p.lambda[j] = vmc->ArrivalRate;
		p.mu[j] = vmc->ServiceRate;
		p.l[j].resize(p.r);
		for (uint32_t h = 0; h < p.r; ++h) {
			p.l[j][h] = vmc->Requirement[h];
		}
		p.revenue[j] = vmc->Revenue;
	}
	return p;
}

void Problem::FromNative(const problem & p) {
	m_resources->Clear();
	for (uint32_t h = 0; h < p.r; ++h) {
		if (p.r_names.size() > 0) {
			m_resources->Add(msclr::interop::marshal_as<String^>(p.r_names[h]));
		} else {
			m_resources->Add("Resource " + (h+1));
		}
	}
	m_serverGroups->Clear();
	for (uint32_t i = 0; i < p.k; ++i) {
		ServerGroup^ g = gcnew ServerGroup();
		if (p.s_names.size() > 0) {
			g->Name = msclr::interop::marshal_as<String^>(p.s_names[i]);
		} else {
			g->Name = "Servers " + (i+1);
		}
		g->Count = p.n[i];
		g->Cost = p.cost[i];
		for (uint32_t h = 0; h < p.r; ++h) {
			g->Capacity->Add(p.c[i][h]);
		}
		m_serverGroups->Add(g);
	}
	m_vmClasses->Clear();
	for (uint32_t j = 0; j < p.m; ++j) {
		VmClass^ v = gcnew VmClass();
		if (p.v_names.size() > 0) {
			v->Name = msclr::interop::marshal_as<String^>(p.v_names[j]);
		} else {
			v->Name = "Virtual Machines " + (j+1);
		}
		v->Revenue = p.revenue[j];
		v->ArrivalRate = p.lambda[j];
		v->ServiceRate = p.mu[j];
		for (uint32_t h = 0; h < p.r; ++h) {
			v->Requirement->Add(p.l[j][h]);
		}
		m_vmClasses->Add(v);
	}
}

void Problem::Save(String^ filename) {
	try {
		std::string fn = msclr::interop::marshal_as<std::string>(filename);
		problem p = ToNative();
		save_problem(fn, p);
	} catch (const problem_save_error & e) {
		throw gcnew ProblemSaveException(msclr::interop::marshal_as<String^>(e.what()));
	}
}

void Problem::Load(String^ filename) {
	try {
		std::string fn = msclr::interop::marshal_as<std::string>(filename);
		problem p = load_problem(fn);
		FromNative(p);
	} catch (const problem_load_error & e) {
		throw gcnew ProblemLoadException(msclr::interop::marshal_as<String^>(e.what()));
	}
}