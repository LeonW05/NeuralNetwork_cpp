#include "matrix.hpp"
#include <cstddef>
#include <random>
#include <stdexcept>
#include <iostream>
#include <string>

Matrix::Matrix(std::size_t rows, std::size_t cols, float value) : _rows(rows), _cols(cols), _data(rows*cols, value) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, const std::vector<float>data) : _rows(rows), _cols(cols), _data(data) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size must be rows*cols");
    }
}

std::size_t Matrix::rows() const {return _rows;}
std::size_t Matrix::cols() const {return _cols;}

float &Matrix::operator()(std::size_t i, std::size_t j) {
    if (i < _rows && j < _cols) {
        return _data[i*_cols + j];
    }
    throw std::out_of_range("Indexes are out of range");
}
const float &Matrix::operator()(std::size_t i, std::size_t j) const {
    if (i < _rows && j < _cols) {
        return _data[i*_cols + j];
    }
    throw std::out_of_range("Indexes are out of range");
}

Matrix Matrix::transpose() const {
    Matrix t(_cols, _rows);

    for (std::size_t i = 0; i < _rows; i++) {
        for (std::size_t j = 0; j < _cols; j++) {
            t(j, i) = (*this)(i, j);
        }
    }

    return t;
}

Matrix Matrix::operator+(const Matrix &o) const {
    if (_rows != o._rows || _cols != o._cols) {
        throw std::invalid_argument("Matrices are not the same dimensions");
    }
    Matrix res(_rows, _cols);

    for (std::size_t i = 0; i < _data.size(); i++) {
        res._data[i] = _data[i] + o._data[i];
    }

    return res;
}

Matrix Matrix::operator-(const Matrix &o) const {
    if (_rows != o._rows || _cols != o._cols) {
        throw std::invalid_argument("Matrices are not the same dimensions");
    }
    Matrix res(_rows, _cols);

    for (std::size_t i = 0; i < _data.size(); i++) {
        res._data[i] = _data[i] - o._data[i];
    }

    return res;
}

Matrix Matrix::hadamard(const Matrix &o) const {
    if (_rows != o._rows || _cols != o._cols) {
        throw std::invalid_argument("Matrices are not the same dimensions");
    }
    Matrix res(_rows, _cols);

    for (std::size_t i = 0; i < _data.size(); i++) {
        res._data[i] = _data[i] * o._data[i];
    }

    return res;
}

Matrix Matrix::scale(float value) const {
    Matrix res(_rows, _cols);

    for (std::size_t i = 0; i < _data.size(); i++) {
        res._data[i] = value * _data[i];
    }

    return res;
}

Matrix Matrix::apply(const std::function<float (float)> &f) const {
    Matrix res(_rows, _cols);

    for (std::size_t i = 0; i < _data.size(); i++) {
        res._data[i] = f(_data[i]);
    }

    return res;
}

Matrix Matrix::matmul(const Matrix &o) const {
    if (_cols != o._rows) {
        throw std::invalid_argument("Matrices can't be multiplied");
    }

    Matrix res(_rows, o._cols);

    for (std::size_t i = 0; i < _rows; i++) {
        for (std::size_t j = 0; j < _cols; j++) {
            for (std::size_t k = 0; k < o._cols; k++) {
                res(i, k) += (*this)(i, j) * o(j, k);
            }
        }
    }

    return res;
}

Matrix Matrix::random(std::size_t rows, std::size_t cols, float limit, std::mt19937 &rng) {
    Matrix res(rows, cols);
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (std::size_t i = 0; i < res._data.size(); i++) {
        res._data[i] = dist(rng);
    }

    return res;
}

void Matrix::print() const {
    std::string string_repr = "[";
    for (std::size_t i = 0; i < _rows; i++) {
        string_repr += "[";
        for (std::size_t j = 0; j < _cols; j++) {
            string_repr += std::to_string((*this)(i, j));
            if (j != _cols - 1) {
                string_repr += ", ";
            }
        }
        string_repr += "]";
        if (i != _rows - 1) {
            string_repr += ", ";
        }
    }
    string_repr += "]";
    std::cout << string_repr << "\n";
}
