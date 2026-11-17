#include <model/model_config.hpp>

#include <stdexcept>

namespace model {

ModelConfig::ModelConfig(const std::string& configFilePath) {
    std::ifstream file(configFilePath);
    if (!file) {
        throw std::runtime_error("Failed to open model config: " + configFilePath);
    }

    const nlohmann::json data = nlohmann::json::parse(file);

    hiddenSize = data.at("hidden_size").get<int>();
    numHiddenLayers = data.at("num_hidden_layers").get<int>();

    headDimension = data.at("head_dim").get<int>();
    numQueryHeads = data.at("num_attention_heads").get<int>();
    // Llama GQA: K and V share the same head count.
    numKeyHeads = data.at("num_key_value_heads").get<int>();
    numValueHeads = numKeyHeads;

    const nlohmann::json& rope = data.at("rope_scaling");
    ropeScalingFactor = rope.at("factor").get<float>();
    ropeHighFrequencyFactor = rope.at("high_freq_factor").get<float>();
    ropeLowFrequencyFactor = rope.at("low_freq_factor").get<float>();
    originalMaxPositionalEmbeddings =
        rope.at("original_max_position_embeddings").get<float>();
    ropeTheta = data.at("rope_theta").get<float>();

    vocabSize = data.at("vocab_size").get<int>();
}

}  // namespace model
