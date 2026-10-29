#include <cstddef>
#include <vector>
#include <functional>
#include <random>

class Matrix {
    private:
        std::size_t _rows;
        std::size_t _cols;
        std::vector<float> _data;
    public:
        Matrix(std::size_t rows , std::size_t cols, float value = 0.0);
        Matrix(std::size_t rows, std::size_t cols, const std::vector<float> data);

        std::size_t rows() const;
        std::size_t cols() const;

        float &operator()(std::size_t i, std::size_t j);
        const float &operator()(std::size_t i, std::size_t j) const;

        Matrix transpose() const;
        Matrix operator+(const Matrix& o) const;
        Matrix operator-(const Matrix& o) const;
        Matrix hadamard(const Matrix& o) const;
        Matrix scale(float value) const;
        Matrix apply(const std::function<float(float)> &f) const;
        Matrix matmul(const Matrix& o) const;

        static Matrix random(std::size_t rows, std::size_t cols, float limit, std::mt19937& rng);

        void print() const;

};
