#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

float forward(float x, float w, float b) {
    return w*x + b;
}

float loss(float y_pred, float y_true) {
    float diff = y_pred - y_true;
    return 0.5 * diff*diff;
}

float mean_loss(const std::vector<float>& xs, const std::vector<float>& ys, float w, float b) {
    float res = 0;

    for (std::size_t i = 0; i < xs.size(); i++) {
        res += loss(forward(xs[i], w, b), ys[i]);
    }

    return res/xs.size();
}

int main() {
    std::vector<float> xs, ys;
    xs.reserve(20);
    ys.reserve(20);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist (-1, 1);

    float w = dist(rng);
    float b = dist(rng);

    for (std::size_t i = 0; i < 20; i++) {
        float x = -1.0 + 2.0f*i / 19.0;
        xs.push_back(x);
        ys.push_back(2*x + 1);
    }

    std::cout << w << " " << b << " " << mean_loss(xs, ys, w, b) << "\n";
    std::cout << mean_loss(xs, ys, 2.0f, 1.0f) << "\n";

    float lr = 0.1;

    for (std::size_t epoch = 0; epoch < 200; epoch++) {
        for (std::size_t i = 0; i < xs.size(); i++) {
            float err = forward(xs[i], w, b) - ys[i];
            w -= lr * err * xs[i];
            b -= lr * err;
        }

        if (epoch % 20 == 0) {
            std::cout << w << " " << b << " " << mean_loss(xs, ys, w, b) << "\n";
        }
    }
    return 0;
}
