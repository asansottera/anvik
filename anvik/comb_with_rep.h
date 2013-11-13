#ifndef ANVIK_COMB_WITH_REP_H
#define ANVIK_COMB_WITH_REP_H

#include <cstdint>
#include <vector>
#include "combinatorics.h"
#include "matrix.h"

/* Class to enumerate and index the:
   - n-combinations of non-necessarily distinct items from a set of cardinality w
   - non-negative integer solution to x_1 + x_2 + ... + x_w = n
   The implementation creates all the states during the init method.
   The get(...) method takes constant time.
 */
class comb_with_rep_v1 {
public:
	void init(uint32_t _n, uint32_t _w);
	uint64_t size() const;
	void get(uint64_t idx, std::vector<uint32_t> & value) const;
	uint64_t find(const std::vector<uint32_t> & value) const;
	const multichoose & multichoose_table() const {
		return mc;
	}
private:
	void enumerate(
		std::vector<uint32_t> state,
		uint32_t missing,
		uint32_t level,
		uint64_t & idx);
private:
	uint32_t n;
	uint32_t w;
	matrix<uint32_t> values;
	multichoose mc;
};

/* Class to enumerate and index the:
   - n-combinations of non-necessarily distinct items from a set of cardinality w
   - non-negative integer solution to x_1 + x_2 + ... + x_w = n
   The implementation requires a constant amount of memory.
 */
class comb_with_rep_v2 {
public:
	void init(uint32_t _n, uint32_t _w);
	uint64_t size() const;
	void get(uint64_t idx, std::vector<uint32_t> & value) const;
	uint64_t find(const std::vector<uint32_t> & value) const;
	const multichoose & multichoose_table() const {
		return mc;
	}
private:
	uint32_t n;
	uint32_t w;
	multichoose mc;
};

#endif