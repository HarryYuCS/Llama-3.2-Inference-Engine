/**

ModelConfig object is from parsed model.config in huggingface model repo,

*/

#include <string>
#include <fstream>
#include <nlohmann/json.hpp>

namespace model {

struct ModelConfig {
    // parses the json config
    ModelConfig(const std::string &configFilePath);

    int hiddenSize; // token embedding dimension

    int numHiddenLayers; // number of transformer blocks

    int headDimension;
    int numQueryHeads;
    int numKeyHeads;
    int numValueHeads;

    // rope : maybe make rope kernel templated?
    float ropeScalingFactor;
    float ropeHighFrequencyFactor;
    float ropeLowFrequencyFactor;
    float originalMaxPositionalEmbeddings;
    float ropeTheta;
    
    int vocabSize;
};

} // namespace model