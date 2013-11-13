// anvik-clr.h

#pragma once

#include "problem.h"

using namespace System;
using namespace System::Collections::Generic;

namespace anvik {

	public ref class ProblemLoadException : public Exception {
	public:
		ProblemLoadException(String^ msg) : Exception(msg) {
		}
	};

	public ref class ProblemSaveException : public Exception {
	public:
		ProblemSaveException(String^ msg) : Exception(msg) {
		}
	};

	public ref class ServerGroup
	{
	public:
		ServerGroup() : 
			m_name(gcnew String("")),
			m_count(1),
			m_cost(0.0f),
			m_capacity(gcnew List<unsigned int>())
		{
		}
		property String^ Name
		{
			String^ get() { return m_name; }
			void set(String^ value) { m_name = value; }
		}
		property unsigned int Count
		{
			unsigned int get() { return m_count; }
			void set(unsigned int value) { m_count = value; }
		}
		property float Cost
		{
			float get() { return m_cost; }
			void set(float value) { m_cost = value; }
		}
		property List<unsigned int>^ Capacity
		{
			List<unsigned int>^ get() { return m_capacity; }
		}
	private:
		String^ m_name;
		unsigned int m_count;
		float m_cost;
		List<unsigned int>^ m_capacity;
	};

	public ref class VmClass
	{
	public:
		VmClass() : 
			m_name(gcnew String("")),
			m_arrivalRate(0.0f),
			m_serviceRate(0.0f),
			m_requirement(gcnew List<unsigned int>())
		{
		}
		property String^ Name
		{
			String^ get() { return m_name; }
			void set(String^ value) { m_name = value; }
		}
		property float ArrivalRate
		{
			float get() { return m_arrivalRate; }
			void set(float value) { m_arrivalRate = value; }
		}
		property float ServiceRate
		{
			float get() { return m_serviceRate; }
			void set(float value) { m_serviceRate = value; }
		}
		property float Revenue
		{
			float get() { return m_revenue; }
			void set(float value) { m_revenue = value; }
		}
		property List<unsigned int>^ Requirement
		{
			List<unsigned int>^ get() { return m_requirement; }
		}
	private:
		String^ m_name;
		float m_arrivalRate;
		float m_serviceRate;
		float m_revenue;
		List<unsigned int>^ m_requirement;
	};

	public ref class Problem
	{
	public:
		Problem() :
			m_resources(gcnew List<String^>()),
			m_serverGroups(gcnew List<ServerGroup^>()),
			m_vmClasses(gcnew List<VmClass^>())
		{
		}
		property List<String^>^ Resources
		{
			List<String^>^ get() { return m_resources; }
		}
		property List<ServerGroup^>^ ServerGroups
		{
			List<ServerGroup^>^ get() { return m_serverGroups; }
		}
		property List<VmClass^>^ VmClasses
		{
			List<VmClass^>^ get() { return m_vmClasses; }
		}
		bool Check(List<String^>^ messages);
		void FromNative(const problem & p);
		problem ToNative();
		void Save(String^ filename);
		void Load(String^ filename);
	private:
		List<String^>^ m_resources;
		List<ServerGroup^>^ m_serverGroups;
		List<VmClass^>^ m_vmClasses;
	};
}
