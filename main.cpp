#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include "matrix.hpp"

static int failed = 0;

#define CHECK(cond)                                           \
    do {                                                      \
        if (cond) std::cout << "[OK]   " << #cond << "\n";    \
        else { std::cout << "[FAIL] " << #cond << "\n"; failed++; } \
    } while (0)

// true si f() lanza una excepcion de tipo Ex
template <typename Ex, typename F>
static bool throws(F f) {
    try { f(); }
    catch (const Ex&) { return true; }
    catch (...) { return false; }
    return false;
}

static bool in_range(const Matrix& m, float lo, float hi) {
    for (std::size_t i = 0; i < m.rows(); i++)
        for (std::size_t j = 0; j < m.cols(); j++)
            if (m(i, j) < lo || m(i, j) > hi) return false;
    return true;
}

static bool equal(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (std::size_t i = 0; i < a.rows(); i++)
        for (std::size_t j = 0; j < a.cols(); j++)
            if (a(i, j) != b(i, j)) return false;
    return true;
}

int main() {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    const Matrix& cA = A;

    std::cout << "--- Constructores, rows/cols, operator() ---\n";
    CHECK(A.rows() == 2);
    CHECK(A.cols() == 3);
    CHECK(A(0, 0) == 1);
    CHECK(A(1, 2) == 6);
    CHECK(cA(1, 0) == 4);
    Matrix Z(2, 2);
    CHECK(Z(0, 0) == 0 && Z(1, 1) == 0);
    Matrix V(2, 2, 7.0f);
    CHECK(V(0, 1) == 7 && V(1, 1) == 7);
    CHECK(throws<std::invalid_argument>([] { Matrix m(2, 3, {1, 2, 3}); (void)m; }));
    CHECK(throws<std::out_of_range>([&] { (void)A(5, 0); }));
    CHECK(throws<std::out_of_range>([&] { (void)A(0, 3); }));
    CHECK(throws<std::out_of_range>([&] { (void)cA(2, 0); }));
    Matrix C = A;
    C(0, 1) = 10;
    CHECK(C(0, 1) == 10);
    CHECK(A(0, 1) == 2);

    std::cout << "\n--- transpose ---\n";
    Matrix At = A.transpose();
    CHECK(At.rows() == 3 && At.cols() == 2);
    CHECK(At(2, 1) == 6);
    CHECK(At(0, 1) == 4);
    CHECK(At(1, 0) == 2);

    std::cout << "\n--- suma y resta ---\n";
    Matrix S = A + A;
    CHECK(S(0, 0) == 2 && S(1, 2) == 12);
    Matrix D = A - A;
    CHECK(D(0, 0) == 0 && D(1, 2) == 0);
    Matrix B24(2, 4), B33(3, 3);
    CHECK(throws<std::invalid_argument>([&] { (void)(A + B24); }));
    CHECK(throws<std::invalid_argument>([&] { (void)(A + B33); }));
    CHECK(throws<std::invalid_argument>([&] { (void)(A - B24); }));
    CHECK(throws<std::invalid_argument>([&] { (void)(A - B33); }));

    std::cout << "\n--- hadamard y scale ---\n";
    Matrix H = A.hadamard(A);
    CHECK(H(0, 1) == 4 && H(1, 2) == 36);
    CHECK(throws<std::invalid_argument>([&] { (void)A.hadamard(B24); }));
    Matrix Sc = A.scale(0.5f);
    CHECK(Sc(0, 1) == 1.0f && Sc(1, 2) == 3.0f);
    CHECK(A(0, 1) == 2);

    std::cout << "\n--- apply ---\n";
    Matrix Z0(1, 1, 0.0f);
    Matrix Sig = Z0.apply([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
    CHECK(std::fabs(Sig(0, 0) - 0.5f) < 1e-6f);
    Matrix Sq = A.apply([](float x) { return x * x; });
    CHECK(Sq(1, 2) == 36 && Sq(0, 0) == 1);
    CHECK(A(1, 2) == 6);

    std::cout << "\n--- matmul ---\n";
    Matrix B(3, 2, {7, 8, 9, 10, 11, 12});
    Matrix P = A.matmul(B);
    CHECK(P.rows() == 2 && P.cols() == 2);
    CHECK(P(0, 0) == 58 && P(0, 1) == 64);
    CHECK(P(1, 0) == 139 && P(1, 1) == 154);
    Matrix G = A.matmul(A.transpose());
    CHECK(G(0, 0) == 14 && G(0, 1) == 32 && G(1, 0) == 32 && G(1, 1) == 77);
    Matrix Q = B.matmul(A);
    CHECK(Q.rows() == 3 && Q.cols() == 3);
    CHECK(Q(0, 0) == 39 && Q(2, 2) == 105);
    CHECK(throws<std::invalid_argument>([&] { (void)A.matmul(A); }));

    std::cout << "\n--- random ---\n";
    std::mt19937 rng(42);
    Matrix R1 = Matrix::random(2, 3, 1.0f, rng);
    Matrix R2 = Matrix::random(2, 3, 1.0f, rng);
    CHECK(R1.rows() == 2 && R1.cols() == 3);
    CHECK(in_range(R1, -1.0f, 1.0f));
    CHECK(in_range(R2, -1.0f, 1.0f));
    CHECK(!equal(R1, R2));

    std::cout << "\n--- print (revisalo a ojo) ---\n";
    std::cout << "A:\n";   A.print();
    std::cout << "At:\n";  At.print();
    std::cout << "A*B:\n"; P.print();
    std::cout << "R1:\n";  R1.print();
    std::cout << "R2:\n";  R2.print();

    std::cout << "\n" << (failed == 0 ? "TODO OK" : "HAY FALLOS") << " (fallos: " << failed << ")\n";
    return failed == 0 ? 0 : 1;
}
