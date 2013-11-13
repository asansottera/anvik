#ifndef ANVIK_PARSE_UTILITIES_H
#define ANVIK_PARSE_UTILITIES_H

#include <string>
#include <sstream>
#include <vector>

inline void trim(std::string & s) {
	s.erase(0, s.find_first_not_of(" \t"));
	s.erase(s.find_last_not_of(" \t")+1);
}

inline void skip_spaces(std::istream & ss)  {
    while (!ss.eof()) {
		char next = ss.peek();
		if (next != ' ' && next != '\t' && next != '\n' && next != '\r') {
			break;
		}
		ss.get();
	}
}

inline void split(std::istringstream & ss, char delim, std::vector<std::string> & items) {
	std::string item;
	items.clear();
	while (std::getline(ss, item, delim)) {
		items.push_back(item);
	}
}


inline void split(const std::string & s, char delim, std::vector<std::string> & items) {
	std::istringstream ss(s);
	split(ss, delim, items);
}

template<class T>
bool try_parse(std::istream & ss, T & result) {
	skip_spaces(ss);
	ss >> result;
	skip_spaces(ss);
	if (!ss.eof()) {
		return false;
	}
	return true;
}

template<>
inline bool try_parse(std::istream & ss, std::string & result) {
	skip_spaces(ss);
	std::getline(ss, result);
	if (!ss.eof()) {
		return false;
	}
	return true;
}

template<class T>
bool try_parse(const std::string & value, T & result) {
	std::istringstream ss(value);
	return try_parse(ss, result);
}

template<class T>
bool try_parse(std::istream & ss, std::vector<T> & result) {
	// open vector
	skip_spaces(ss);
	char open_square_bracket = ss.get();
	if (open_square_bracket != '[') {
		return false;
	}
	// vector elements
	std::string items;
	std::getline(ss, items, ']');
	std::vector<std::string> splitted;
	split(items, ',', splitted);
	result.reserve(splitted.size());
	typedef std::vector<std::string>::iterator iter;
	for (iter it = splitted.begin(); it < splitted.end(); ++it) {
		std::istringstream ssit(*it);
		T parsed;
		if (!try_parse(ssit, parsed)) {
			return false;
		}
		result.push_back(parsed);
	}
	// close vector
	skip_spaces(ss);
	if (!ss.eof()) {
		return false;
	}
	// no errors!
	return true;
}

template<class T>
bool try_parse(const std::string & value, std::vector<T> & result) {
	std::istringstream ss(value);
	return try_parse(ss, result);
}

template<class T>
bool try_parse(std::istream & ss, std::vector<std::vector<T>> & result) {
	// open vector
	skip_spaces(ss);
	if (ss.get() != '{') {
		return false;
	}
	// vector elements
	std::string items;
	std::getline(ss, items, '}');
	std::vector<std::string> splitted;
	split(items, ';', splitted);
	result.reserve(splitted.size());
	typedef std::vector<std::string>::iterator iter;
	for (iter it = splitted.begin(); it < splitted.end(); ++it) {
		std::istringstream ssit(*it);
		std::vector<T> parsed;
		if (!try_parse(ssit, parsed)) {
			return false;
		}
		result.push_back(parsed);
	}
	// close vector
	skip_spaces(ss);
	if (!ss.eof()) {
		return false;
	}
	// no errors!
	return true;
}

template<class T>
bool try_parse(const std::string & value, std::vector<std::vector<T>> & result) {
	std::istringstream ss(value);
	return try_parse(ss, result);
}

#endif