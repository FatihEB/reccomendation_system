#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {
constexpr int dimensions = 32;
using Factors = std::array<float, dimensions>;

struct Rating {
    std::uint32_t user;
    std::uint32_t item;
    float stars;
};
static_assert(sizeof(Rating) == 12);

struct Entity {
    float bias = 0;
    Factors factors;
};
static_assert(sizeof(Entity) == (dimensions + 1) * sizeof(float));

void initialize(std::vector<Entity>& entities, std::mt19937& random) {
    std::normal_distribution<float> distribution(0.f, 0.04f);
    for (auto& entity : entities) {
        for (auto& factor : entity.factors) {
            factor = distribution(random);
        }
    }
}
}

int main(int argc, char** argv) {
    if (argc != 9) {
        return 1;
    }
    const std::size_t rows = std::stoull(argv[3]);
    const unsigned users = std::stoul(argv[4]);
    const unsigned items = std::stoul(argv[5]);
    const unsigned epochs = std::stoul(argv[6]);
    std::mt19937 random(std::stoul(argv[7]));
    const float global = std::stof(argv[8]);
    std::vector<Rating> ratings(rows);
    std::ifstream input(argv[1], std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(ratings.data()), rows * sizeof(Rating))) {
        std::cerr << "Could not read the ratings cache\n";
        return 1;
    }
    std::vector<std::size_t> start(users + 1);
    std::size_t cursor = 0;
    for (unsigned user = 0; user < users; ++user) {
        start[user] = cursor;
        while (cursor < rows && ratings[cursor].user == user) {
            if (ratings[cursor].item >= items) {
                return 1;
            }
            ++cursor;
        }
    }
    if (cursor != rows) {
        std::cerr << "Expected MovieLens ratings sorted by user\n";
        return 1;
    }
    start[users] = rows;
    std::vector<Entity> people(users), films(items);
    initialize(people, random);
    initialize(films, random);
    std::vector<unsigned> order(users);
    std::iota(order.begin(), order.end(), 0);
    for (unsigned epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), random);
        const float step = 0.014f / (1.f + 0.12f * epoch);
        const float shrinkage = step * 0.055f;
        double squared_error = 0;
        for (const unsigned user : order) {
            auto& person = people[user];
            for (std::size_t row = start[user]; row < start[user + 1]; ++row) {
                const auto& rating = ratings[row];
                auto& film = films[rating.item];
                float predicted = global + person.bias + film.bias;
                for (int k = 0; k < dimensions; ++k) {
                    predicted += person.factors[k] * film.factors[k];
                }
                const float error = rating.stars - predicted;
                squared_error += error * error;
                const float gradient = step * error;
                person.bias += gradient - shrinkage * person.bias;
                film.bias += gradient - shrinkage * film.bias;
                for (int k = 0; k < dimensions; ++k) {
                    const float p = person.factors[k], q = film.factors[k];
                    person.factors[k] += gradient * q - shrinkage * p;
                    film.factors[k] += gradient * p - shrinkage * q;
                }
            }
        }
        std::cout << "Epoch " << epoch + 1 << " online training RMSE " << std::sqrt(squared_error / rows) << std::endl;
    }
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char*>(films.data()), films.size() * sizeof(Entity));
    return output ? 0 : 1;
}
