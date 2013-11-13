#include "comb_with_rep.h"

#include <cassert>
#include <limits>
#include <stdexcept>
#include <sstream>

void comb_with_rep_v1::init(uint32_t _n, uint32_t _w) {
	n = _n;
	w = _w;
	mc.init(n, w);
	uint64_t size = mc(n, w);
	values.resize(w, size);
	// enumerate states
	std::vector<uint32_t> empty(w);
	uint64_t idx = 0;
	enumerate(empty, n, 0, idx);
}

void comb_with_rep_v1::enumerate(
	std::vector<uint32_t> value,
	uint32_t missing,
	uint32_t level,
	uint64_t & idx) {
	// put all remaining items in the current bin
	value[level] = missing;
	values.set_column(idx, value);
	++idx;
	// reduce number of items in the current bin
	if (level == w-1) {
		return;
	}
	for (int32_t z = missing-1; z >= 0; --z) {
		value[level] = z;
		enumerate(value, missing-z, level+1, idx);
	}
}

uint64_t comb_with_rep_v1::size() const {
	return values.cols();
}

void comb_with_rep_v1::get(uint64_t idx, std::vector<uint32_t> & value) const {
	values.get_column(idx, value);
}

uint64_t comb_with_rep_v1::find(const std::vector<uint32_t> & value) const {
	assert(value.size() == w);
	uint64_t found = 0;
	uint32_t remaining = n;
	for (uint32_t l = 0; l < w; ++l) {
		// remaininb bins
		uint32_t ra = w-l-1;
		for (uint32_t h = remaining; h > value[l]; --h) {
			// remaining items
			uint32_t rs = remaining - h;
			found += mc(rs, ra);
		}
		remaining -= value[l];
	}
	return found;
}

void comb_with_rep_v2::init(uint32_t _n, uint32_t _w) {
	n = _n;
	w = _w;
	mc.init(n, w);
}

uint64_t comb_with_rep_v2::size() const {
	return mc(n, w);
}

void comb_with_rep_v2::get(uint64_t idx, std::vector<uint32_t> & value) const {
	value.resize(w);
	std::fill(value.begin(), value.end(), 0);
	value[0] = n;
	uint32_t remaining = 0;
	uint32_t bin = 0;
	uint64_t current = 0;
	while (current != idx) {
		value[bin] -= 1;
		remaining += 1;
		uint64_t next = current + mc(remaining, w-(bin+1));
		if (next > idx) {
			bin += 1;
			value[bin] = remaining;
			remaining = 0;
			current += 1;
		} else if (next < idx) {
			current = next;
		} else {
			current = next;
			value[w-1] = remaining;
		}
	}
}

uint64_t comb_with_rep_v2::find(const std::vector<uint32_t> & value) const {
	assert(value.size() == w);
	uint64_t found = 0;
	uint32_t remaining = n;
	for (uint32_t l = 0; l < w; ++l) {
		// update remamining items
		remaining -= value[l];
		// consider remaining bins
		uint32_t ra = w-l-1;
		for (uint32_t rs = 0; rs < remaining; ++rs) {
			found += mc(rs, ra);
		}
		if (remaining == 0) {
			break;
		}
	}
	return found;
}