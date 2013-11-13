#ifndef ANVIK_MATRIX_H
#define ANVIK_MATRIX_H

#include <cstdint>
#include <memory>
#include <cstring>
#include <cassert>

template<class T>
class matrix {
private:
	T * m_memory;
	uint64_t m_rows;
	uint64_t m_cols;
public:
	matrix() {
		m_memory = 0;
		m_rows = 0;
		m_cols = 0;
	}
	matrix(uint64_t rows, uint64_t cols) {
		m_memory = new T[rows * cols];
		m_rows = rows;
		m_cols = cols;
	}
	~matrix() {
		delete[] m_memory;
	}
	void resize(uint64_t rows, uint64_t cols) {
		delete[] m_memory;
		m_memory = new T[rows * cols];
		m_rows = rows;
		m_cols = cols;
	}
	const T * raw() const {
		return m_memory;
	}
	T * raw() {
		return m_memory;
	}
	uint64_t size() const {
		return m_rows * m_cols;
	}
	uint64_t rows() const {
		return m_rows;
	}
	uint64_t cols() const {
		return m_cols;
	}
	T operator()(uint64_t row, uint64_t col) const {
		return m_memory[col * m_rows + row];
	}
	T & operator()(uint64_t row, uint64_t col) {
		return m_memory[col * m_rows + row];
	}
	void get_column(uint64_t col, std::vector<T> & value) const {
		value.resize(m_rows);
		std::memcpy(&value[0], m_memory + col * m_rows, m_rows * sizeof(T));
	}
	void set_column(uint64_t col, const std::vector<T> & value) {
		std::memcpy(m_memory + col * m_rows, &value[0], m_rows * sizeof(T));
	}
	void get_row(uint64_t row, std::vector<T> & value) const  {
		value.resize(m_cols);
		for (uint64_t col = 0; col < m_cols; ++col) {
			value(col) = (*this)(row,col);
		}
	}
	void set_row(uint64_t row, const std::vector<T> & value) {
		for (uint64_t col = 0; col < m_cols; ++col) {
			(*this)(row,col) = value(col);
		}
	}
};

#endif
