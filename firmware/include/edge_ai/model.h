#ifndef EDGE_AI_MODEL_H
#define EDGE_AI_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#define EDGE_AI_CHANNEL_COUNT 3
#define EDGE_AI_WINDOW_SAMPLES 64
#define EDGE_AI_FEATURE_OUTPUT_COUNT 18

int edge_ai_extract_features(
    const float samples[EDGE_AI_CHANNEL_COUNT][EDGE_AI_WINDOW_SAMPLES],
    float features[EDGE_AI_FEATURE_OUTPUT_COUNT]);

int edge_ai_predict_features(const float features[EDGE_AI_FEATURE_OUTPUT_COUNT]);

int edge_ai_predict_samples(
    const float samples[EDGE_AI_CHANNEL_COUNT][EDGE_AI_WINDOW_SAMPLES]);

#ifdef __cplusplus
}
#endif

#endif
