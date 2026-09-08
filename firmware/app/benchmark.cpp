#include "edge_ai/model.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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

double percentile(const std::vector<double>& sorted, double fraction) {
    const auto index = static_cast<std::size_t>(fraction * static_cast<double>(sorted.size() - 1));
    return sorted[index];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("usage: edge_ai_benchmark <reference_vectors.csv> <output.json>");
        }
        std::ifstream input(argv[1]);
        if (!input.good()) {
            throw std::runtime_error("could not open reference vectors");
        }
        std::string header;
        std::string line;
        std::getline(input, header);
        std::getline(input, line);
        const auto fields = split_csv(line);
        float samples[EDGE_AI_CHANNEL_COUNT][EDGE_AI_WINDOW_SAMPLES]{};
        constexpr std::size_t metadata_count = 5;
        constexpr std::size_t sample_count =
            EDGE_AI_CHANNEL_COUNT * EDGE_AI_WINDOW_SAMPLES;
        if (fields.size() < metadata_count + sample_count) {
            throw std::runtime_error("reference vector is incomplete");
        }
        for (std::size_t index = 0; index < sample_count; ++index) {
            samples[index / EDGE_AI_WINDOW_SAMPLES][index % EDGE_AI_WINDOW_SAMPLES] =
                std::stof(fields[metadata_count + index]);
        }

        volatile int prediction_checksum = 0;
        for (int index = 0; index < 1000; ++index) {
            prediction_checksum = prediction_checksum + edge_ai_predict_samples(samples);
        }

        constexpr int iterations = 20000;
        constexpr double host_budget_us = 1000.0;
        std::vector<double> durations;
        durations.reserve(iterations);
        int deadline_misses = 0;
        for (int index = 0; index < iterations; ++index) {
            const auto start = std::chrono::steady_clock::now();
            prediction_checksum = prediction_checksum + edge_ai_predict_samples(samples);
            const auto end = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double, std::micro>(end - start).count();
            durations.push_back(elapsed);
            if (elapsed > host_budget_us) {
                ++deadline_misses;
            }
        }
        std::sort(durations.begin(), durations.end());
        const double mean =
            std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();

        std::ofstream output(argv[2]);
        if (!output.good()) {
            throw std::runtime_error("could not create benchmark output");
        }
        output << std::fixed << std::setprecision(6)
               << "{\n"
               << "  \"scope\": \"host timing only; not MCU, RTOS, or hard-real-time evidence\",\n"
               << "  \"iterations\": " << iterations << ",\n"
               << "  \"host_budget_us\": " << host_budget_us << ",\n"
               << "  \"deadline_misses\": " << deadline_misses << ",\n"
               << "  \"mean_us\": " << mean << ",\n"
               << "  \"p50_us\": " << percentile(durations, 0.50) << ",\n"
               << "  \"p95_us\": " << percentile(durations, 0.95) << ",\n"
               << "  \"p99_us\": " << percentile(durations, 0.99) << ",\n"
               << "  \"max_us\": " << durations.back() << ",\n"
               << "  \"prediction_checksum\": " << prediction_checksum << "\n"
               << "}\n";
        std::cout << "host benchmark: mean=" << mean
                  << " us, p95=" << percentile(durations, 0.95)
                  << " us, misses=" << deadline_misses << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
