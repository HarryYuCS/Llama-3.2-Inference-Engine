#include <model/weight_store.hpp>
#include <model/model_config.hpp>

#include <core/cuda_check.hpp>
#include <core/tensor.hpp>

#include <stdexcept>
#include <unordered_map>

/**

TODO : parse safetensors, initialize tensors for each layer

*/

namespace model {

WeightStore::WeightStore(const ModelConfig &cfg) {

}

const core::DeviceTensor &WeightStore::get(const std::string &name) const {
    if (!tensors_.contains(name)) {
        throw std::invalid_argument("No tensor of that name exists in the weight store.");
    }

    return tensors_.at(name);
}

void WeightStore::parseSafetensors_(const std::string &path) {

}

}