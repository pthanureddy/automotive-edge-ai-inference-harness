#include "edge_ai/model.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: edge_ai_tests <reference_vectors.csv>");
        std::ifstream input(argv[1]);
        require(input.good(), "could not open reference vectors");
        std::string line;
        require(static_cast<bool>(std::getline(input, line)), "missing CSV header");

        int vector_count = 0;
        while (std::getline(input, line)) {
            const auto fields = split_csv(line);
            constexpr std::size_t metadata_count = 5;
            constexpr std::size_t sample_count =
                EDGE_AI_CHANNEL_COUNT * EDGE_AI_WINDOW_SAMPLES;
            require(
                fields.size() == metadata_count + sample_count + EDGE_AI_FEATURE_OUTPUT_COUNT,
                "unexpected reference vector width");

            const int expected_prediction = std::stoi(fields[2]);
            float samples[EDGE_AI_CHANNEL_COUNT][EDGE_AI_WINDOW_SAMPLES]{};
            float expected_features[EDGE_AI_FEATURE_OUTPUT_COUNT]{};
            for (std::size_t index = 0; index < sample_count; ++index) {
                samples[index / EDGE_AI_WINDOW_SAMPLES][index % EDGE_AI_WINDOW_SAMPLES] =
                    std::stof(fields[metadata_count + index]);
            }
            for (std::size_t index = 0; index < EDGE_AI_FEATURE_OUTPUT_COUNT; ++index) {
                expected_features[index] =
                    std::stof(fields[metadata_count + sample_count + index]);
            }

            float actual_features[EDGE_AI_FEATURE_OUTPUT_COUNT]{};
            require(edge_ai_extract_features(samples, actual_features) == 0, "feature status");
            for (int index = 0; index < EDGE_AI_FEATURE_OUTPUT_COUNT; ++index) {
                const float tolerance = 2.5e-4F + 2.5e-3F * std::abs(expected_features[index]);
                require(
                    std::abs(actual_features[index] - expected_features[index]) <= tolerance,
                    "Python/C feature mismatch at index " + std::to_string(index));
            }
            require(
                edge_ai_predict_features(actual_features) == expected_prediction,
                "C/Python quantized prediction mismatch");
            require(
                edge_ai_predict_samples(samples) == expected_prediction,
                "end-to-end C prediction mismatch");
            ++vector_count;
        }

        require(vector_count == 15, "expected 15 reference vectors");
        float invalid[EDGE_AI_FEATURE_OUTPUT_COUNT]{};
        invalid[3] = NAN;
        require(edge_ai_predict_features(invalid) == -2, "non-finite input must fail");
        std::cout << vector_count << " C/C++ reference vectors passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
