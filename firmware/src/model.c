#include "edge_ai/model.h"

#include "model_data.h"

#include <math.h>
#include <stdint.h>

#define EDGE_AI_PI 3.14159265358979323846f

static int8_t quantize_symmetric(float value, float scale, int lower, int upper) {
    long result = lroundf(value / scale);
    if (result < lower) {
        result = lower;
    }
    if (result > upper) {
        result = upper;
    }
    return (int8_t)result;
}

int edge_ai_extract_features(
    const float samples[EDGE_AI_CHANNEL_COUNT][EDGE_AI_WINDOW_SAMPLES],
    float features[EDGE_AI_FEATURE_OUTPUT_COUNT]) {
    if (samples == 0 || features == 0) {
        return -1;
    }

    for (int axis = 0; axis < EDGE_AI_CHANNEL_COUNT; ++axis) {
        float mean = 0.0f;
        for (int sample = 0; sample < EDGE_AI_WINDOW_SAMPLES; ++sample) {
            if (!isfinite(samples[axis][sample])) {
                return -2;
            }
            mean += samples[axis][sample];
        }
        mean /= (float)EDGE_AI_WINDOW_SAMPLES;

        float squared_sum = 0.0f;
        float peak = 0.0f;
        int crossings = 0;
        for (int sample = 0; sample < EDGE_AI_WINDOW_SAMPLES; ++sample) {
            const float value = samples[axis][sample];
            const float centered = value - mean;
            squared_sum += value * value;
            const float magnitude = fabsf(centered);
            if (magnitude > peak) {
                peak = magnitude;
            }
            if (sample > 0) {
                const float previous = samples[axis][sample - 1] - mean;
                if ((previous < 0.0f) != (centered < 0.0f)) {
                    ++crossings;
                }
            }
        }

        float low_power = 0.0f;
        float high_power = 0.0f;
        for (int bin = 1; bin <= 12; ++bin) {
            float real = 0.0f;
            float imaginary = 0.0f;
            for (int sample = 0; sample < EDGE_AI_WINDOW_SAMPLES; ++sample) {
                const float centered = samples[axis][sample] - mean;
                const float angle = -2.0f * EDGE_AI_PI * (float)(bin * sample)
                    / (float)EDGE_AI_WINDOW_SAMPLES;
                real += centered * cosf(angle);
                imaginary += centered * sinf(angle);
            }
            const float power = (real * real + imaginary * imaginary)
                / (float)(EDGE_AI_WINDOW_SAMPLES * EDGE_AI_WINDOW_SAMPLES);
            if (bin <= 3) {
                low_power += power;
            } else {
                high_power += power;
            }
        }

        const int offset = axis * 6;
        features[offset] = mean;
        features[offset + 1] = sqrtf(squared_sum / (float)EDGE_AI_WINDOW_SAMPLES);
        features[offset + 2] = peak;
        features[offset + 3] = (float)crossings / (float)(EDGE_AI_WINDOW_SAMPLES - 1);
        features[offset + 4] = low_power;
        features[offset + 5] = high_power;
    }
    return 0;
}

int edge_ai_predict_features(const float features[EDGE_AI_FEATURE_OUTPUT_COUNT]) {
    if (features == 0) {
        return -1;
    }

    int8_t input[EDGE_AI_FEATURE_COUNT];
    for (int index = 0; index < EDGE_AI_FEATURE_COUNT; ++index) {
        if (!isfinite(features[index])) {
            return -2;
        }
        const float standardized =
            (features[index] - EDGE_AI_SCALER_MEAN[index]) / EDGE_AI_SCALER_SCALE[index];
        input[index] = quantize_symmetric(standardized, EDGE_AI_INPUT_SCALE, -127, 127);
    }

    int8_t hidden[EDGE_AI_HIDDEN_COUNT];
    for (int hidden_index = 0; hidden_index < EDGE_AI_HIDDEN_COUNT; ++hidden_index) {
        int32_t accumulator = 0;
        for (int input_index = 0; input_index < EDGE_AI_FEATURE_COUNT; ++input_index) {
            accumulator += (int32_t)input[input_index]
                * (int32_t)EDGE_AI_WEIGHT1[input_index * EDGE_AI_HIDDEN_COUNT + hidden_index];
        }
        float activation = (float)accumulator * EDGE_AI_INPUT_SCALE * EDGE_AI_WEIGHT1_SCALE
            + EDGE_AI_BIAS1[hidden_index];
        if (activation < 0.0f) {
            activation = 0.0f;
        }
        hidden[hidden_index] = quantize_symmetric(activation, EDGE_AI_HIDDEN_SCALE, 0, 127);
    }

    float best_logit = -INFINITY;
    int best_class = -1;
    for (int class_index = 0; class_index < EDGE_AI_CLASS_COUNT; ++class_index) {
        int32_t accumulator = 0;
        for (int hidden_index = 0; hidden_index < EDGE_AI_HIDDEN_COUNT; ++hidden_index) {
            accumulator += (int32_t)hidden[hidden_index]
                * (int32_t)EDGE_AI_WEIGHT2[hidden_index * EDGE_AI_CLASS_COUNT + class_index];
        }
        const float logit = (float)accumulator * EDGE_AI_HIDDEN_SCALE * EDGE_AI_WEIGHT2_SCALE
            + EDGE_AI_BIAS2[class_index];
        if (logit > best_logit) {
            best_logit = logit;
            best_class = class_index;
        }
    }
    return (int)EDGE_AI_CLASSES[best_class];
}

int edge_ai_predict_samples(
    const float samples[EDGE_AI_CHANNEL_COUNT][EDGE_AI_WINDOW_SAMPLES]) {
    float features[EDGE_AI_FEATURE_OUTPUT_COUNT];
    const int status = edge_ai_extract_features(samples, features);
    if (status != 0) {
        return status;
    }
    return edge_ai_predict_features(features);
}
