#ifndef ANVIK_PRINT_UTILITIES_H
#define ANVIK_PRINT_UTILITIES_H

#include <iostream>
#include <vector>

template<class T>
std::ostream & operator<<(std::ostream & out, const std::vector<T> & v) {
	out << "[";
	if (v.size() > 0) {
		for (std::size_t l = 0; l < v.size()-1; ++l) {
			out << v[l] << ", ";
		}
		out << v[v.size()-1] << "]";
	} else {
		out << "]";
	}
	return out;
}

template<class T>
std::ostream & operator<<(std::ostream & out, const std::vector<std::vector<T>> & v) {
	out << "{";
	if (v.size() > 0) {
		for (std::size_t i = 0; i < v.size()-1; ++i) {
			out << v[i] << "; ";
		}	
		out << v[v.size()-1];
	}
	out << "}";
	return out;
}

#endif