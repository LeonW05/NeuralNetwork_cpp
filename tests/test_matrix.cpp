#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include "../matrix.hpp"

namespace {

// A = [[1,2,3],[4,5,6]]  (2x3)
Matrix make_A() { return Matrix(2, 3, {1, 2, 3, 4, 5, 6}); }
// B = [[7,8],[9,10],[11,12]]  (3x2)
Matrix make_B() { return Matrix(3, 2, {7, 8, 9, 10, 11, 12}); }

// Comprueba dimensiones y todos los elementos
void expect_matrix_eq(const Matrix& got, const Matrix& want) {
    ASSERT_EQ(got.rows(), want.rows());
    ASSERT_EQ(got.cols(), want.cols());
    for (std::size_t i = 0; i < got.rows(); i++)
        for (std::size_t j = 0; j < got.cols(); j++)
            EXPECT_FLOAT_EQ(got(i, j), want(i, j)) << "en (" << i << ", " << j << ")";
}

void expect_in_range(const Matrix& m, float lo, float hi) {
    for (std::size_t i = 0; i < m.rows(); i++)
        for (std::size_t j = 0; j < m.cols(); j++) {
            EXPECT_GE(m(i, j), lo);
            EXPECT_LE(m(i, j), hi);
        }
}

}  // namespace

// ---------- Constructores ----------

TEST(MatrixConstructor, DefaultValueIsZero) {
    Matrix Z(2, 2);
    expect_matrix_eq(Z, Matrix(2, 2, {0, 0, 0, 0}));
}

TEST(MatrixConstructor, FilledWithValue) {
    Matrix V(2, 3, 7.0f);
    EXPECT_EQ(V.rows(), 2u);
    EXPECT_EQ(V.cols(), 3u);
    expect_matrix_eq(V, Matrix(2, 3, {7, 7, 7, 7, 7, 7}));
}

TEST(MatrixConstructor, FromDataKeepsRowMajorOrder) {
    Matrix A = make_A();
    EXPECT_EQ(A.rows(), 2u);
    EXPECT_EQ(A.cols(), 3u);
    EXPECT_FLOAT_EQ(A(0, 0), 1);
    EXPECT_FLOAT_EQ(A(0, 2), 3);
    EXPECT_FLOAT_EQ(A(1, 0), 4);
    EXPECT_FLOAT_EQ(A(1, 2), 6);
}

TEST(MatrixConstructor, WrongDataSizeThrows) {
    EXPECT_THROW({ Matrix m(2, 3, {1, 2, 3}); (void)m; }, std::invalid_argument);
    EXPECT_THROW({ Matrix m(1, 2, {1, 2, 3}); (void)m; }, std::invalid_argument);
}

// ---------- Acceso ----------

TEST(MatrixAccess, WriteThroughReference) {
    Matrix A = make_A();
    A(0, 1) = 10;
    EXPECT_FLOAT_EQ(A(0, 1), 10);
}

TEST(MatrixAccess, CopyIsIndependent) {
    Matrix A = make_A();
    Matrix C = A;
    C(0, 1) = 10;
    EXPECT_FLOAT_EQ(C(0, 1), 10);
    EXPECT_FLOAT_EQ(A(0, 1), 2);
}

TEST(MatrixAccess, ConstAccess) {
    const Matrix A = make_A();
    EXPECT_FLOAT_EQ(A(1, 0), 4);
}

TEST(MatrixAccess, OutOfRangeThrows) {
    Matrix A = make_A();
    const Matrix& cA = A;
    EXPECT_THROW((void)A(5, 0), std::out_of_range);
    EXPECT_THROW((void)A(0, 3), std::out_of_range);
    EXPECT_THROW((void)A(2, 0), std::out_of_range);
    EXPECT_THROW((void)cA(5, 0), std::out_of_range);
    EXPECT_THROW((void)cA(0, 3), std::out_of_range);
}

// ---------- Transpose ----------

TEST(MatrixTranspose, SwapsDimensionsAndElements) {
    Matrix At = make_A().transpose();
    expect_matrix_eq(At, Matrix(3, 2, {1, 4, 2, 5, 3, 6}));
}

TEST(MatrixTranspose, TwiceIsIdentity) {
    Matrix A = make_A();
    expect_matrix_eq(A.transpose().transpose(), A);
}

// ---------- Suma y resta ----------

TEST(MatrixAdd, AddsElementWise) {
    Matrix A = make_A();
    expect_matrix_eq(A + A, Matrix(2, 3, {2, 4, 6, 8, 10, 12}));
}

TEST(MatrixSub, SubtractsElementWise) {
    Matrix A = make_A();
    expect_matrix_eq(A - A, Matrix(2, 3, 0.0f));
    expect_matrix_eq(A + A - A, A);
}

TEST(MatrixAddSub, DimensionMismatchThrows) {
    Matrix A = make_A();
    Matrix B24(2, 4), B33(3, 3);
    EXPECT_THROW((void)(A + B24), std::invalid_argument);
    EXPECT_THROW((void)(A + B33), std::invalid_argument);
    EXPECT_THROW((void)(A - B24), std::invalid_argument);
    EXPECT_THROW((void)(A - B33), std::invalid_argument);
}

TEST(MatrixAddSub, DoNotModifyOperands) {
    Matrix A = make_A();
    Matrix B = A;
    (void)(A + B);
    (void)(A - B);
    expect_matrix_eq(A, make_A());
}

// ---------- Hadamard, scale ----------

TEST(MatrixHadamard, MultipliesElementWise) {
    Matrix A = make_A();
    expect_matrix_eq(A.hadamard(A), Matrix(2, 3, {1, 4, 9, 16, 25, 36}));
}

TEST(MatrixHadamard, DimensionMismatchThrows) {
    Matrix A = make_A();
    EXPECT_THROW((void)A.hadamard(Matrix(2, 4)), std::invalid_argument);
    EXPECT_THROW((void)A.hadamard(Matrix(3, 2)), std::invalid_argument);
}

TEST(MatrixScale, MultipliesByScalar) {
    Matrix A = make_A();
    expect_matrix_eq(A.scale(0.5f), Matrix(2, 3, {0.5f, 1, 1.5f, 2, 2.5f, 3}));
    expect_matrix_eq(A.scale(0.0f), Matrix(2, 3, 0.0f));
    expect_matrix_eq(A.scale(-1.0f), Matrix(2, 3, {-1, -2, -3, -4, -5, -6}));
}

TEST(MatrixScale, DoesNotModifyOriginal) {
    Matrix A = make_A();
    (void)A.scale(3.0f);
    expect_matrix_eq(A, make_A());
}

// ---------- Apply ----------

TEST(MatrixApply, Sigmoid) {
    Matrix Z(1, 3, {0.0f, 100.0f, -100.0f});
    Matrix S = Z.apply([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
    EXPECT_NEAR(S(0, 0), 0.5f, 1e-6f);
    EXPECT_NEAR(S(0, 1), 1.0f, 1e-6f);
    EXPECT_NEAR(S(0, 2), 0.0f, 1e-6f);
}

TEST(MatrixApply, Square) {
    Matrix A = make_A();
    expect_matrix_eq(A.apply([](float x) { return x * x; }),
                     Matrix(2, 3, {1, 4, 9, 16, 25, 36}));
}

TEST(MatrixApply, ReLU) {
    Matrix Z(1, 4, {-2, -0.5f, 0, 3});
    expect_matrix_eq(Z.apply([](float x) { return x > 0 ? x : 0.0f; }),
                     Matrix(1, 4, {0, 0, 0, 3}));
}

TEST(MatrixApply, DoesNotModifyOriginal) {
    Matrix A = make_A();
    (void)A.apply([](float x) { return x + 100; });
    expect_matrix_eq(A, make_A());
}

// ---------- matmul ----------

TEST(MatrixMatMul, KnownProduct) {
    expect_matrix_eq(make_A().matmul(make_B()), Matrix(2, 2, {58, 64, 139, 154}));
}

TEST(MatrixMatMul, DifferentShapeProduct) {
    expect_matrix_eq(make_B().matmul(make_A()),
                     Matrix(3, 3, {39, 54, 69, 49, 68, 87, 59, 82, 105}));
}

TEST(MatrixMatMul, TimesTranspose) {
    Matrix A = make_A();
    expect_matrix_eq(A.matmul(A.transpose()), Matrix(2, 2, {14, 32, 32, 77}));
}

TEST(MatrixMatMul, IdentityLeavesMatrixUnchanged) {
    Matrix A = make_A();
    Matrix I2(2, 2, {1, 0, 0, 1});
    Matrix I3(3, 3, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    expect_matrix_eq(I2.matmul(A), A);
    expect_matrix_eq(A.matmul(I3), A);
}

TEST(MatrixMatMul, ColumnVector) {
    // (2x3) * (3x1) -> (2x1)
    Matrix x(3, 1, {1, 0, -1});
    expect_matrix_eq(make_A().matmul(x), Matrix(2, 1, {-2, -2}));
}

TEST(MatrixMatMul, IncompatibleDimensionsThrow) {
    Matrix A = make_A();
    EXPECT_THROW((void)A.matmul(A), std::invalid_argument);
    EXPECT_THROW((void)A.matmul(Matrix(2, 2)), std::invalid_argument);
}

// ---------- random ----------

TEST(MatrixRandom, ValuesWithinLimit) {
    std::mt19937 rng(42);
    Matrix R = Matrix::random(20, 20, 0.5f, rng);
    EXPECT_EQ(R.rows(), 20u);
    EXPECT_EQ(R.cols(), 20u);
    expect_in_range(R, -0.5f, 0.5f);
}

TEST(MatrixRandom, ConsecutiveCallsDiffer) {
    std::mt19937 rng(42);
    Matrix R1 = Matrix::random(2, 3, 1.0f, rng);
    Matrix R2 = Matrix::random(2, 3, 1.0f, rng);
    bool any_different = false;
    for (std::size_t i = 0; i < 2; i++)
        for (std::size_t j = 0; j < 3; j++)
            if (R1(i, j) != R2(i, j)) any_different = true;
    EXPECT_TRUE(any_different);
}

TEST(MatrixRandom, SameSeedSameMatrix) {
    std::mt19937 rng1(7), rng2(7);
    expect_matrix_eq(Matrix::random(3, 3, 1.0f, rng1), Matrix::random(3, 3, 1.0f, rng2));
}
